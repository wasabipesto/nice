// CUDA kernels for nice number checking.
//
// These kernels are compiled at runtime by NVRTC (see client_process_gpu.rs),
// once per (base, mode) pair, with all base-dependent values baked in as
// preprocessor defines. This is the GPU analog of the CPU's const-generic
// dispatch: every division below has a compile-time-constant divisor, so the
// compiler strength-reduces it to multiply-high sequences. There is no
// hardware division anywhere in the hot path.
//
// Defines injected by the host (client_process_gpu.rs):
//   BASE          - the numeric base
//   N_LIMBS       - u32 limbs needed to hold the largest n in this base's range
//   CHUNK_DIGITS  - digits extracted per radix chunk (largest E with BASE^E < 2^31)
//   CHUNK_DIV     - BASE^CHUNK_DIGITS
// For the niceonly kernel (define NICEONLY):
//   STRIDE_M      - stride filter modulus M = (BASE-1) * BASE^k
//   STRIDE_R      - number of valid residues mod M
//   POW64_MOD_M   - 2^64 mod STRIDE_M (for u128 mod M)
// Optionally for the niceonly kernel (define PREFILTER; host enables it only
// when n^2 and n^3 are guaranteed at least PRE_DIGITS digits each):
//   PRE_DIGITS    - digits checked per value in the modular prefilter
//   PRE_MOD       - BASE^PRE_DIGITS (u64, <= 2^48)
//   POW64_MOD_PRE - 2^64 mod PRE_MOD
// For the detailed kernel (define DETAILED):
//   NEAR_MISS_CUTOFF - num_uniques above which a number is reported
//
// Numbers n are passed as (lo, hi) u64 pairs (CUDA has no native u128 and we
// deliberately avoid __int128 for NVRTC compatibility). All wide arithmetic
// is done in u32 limbs with u64 accumulators.
//
// Kernel work assignment:
//   niceonly_ranges_kernel: one warp per MSD-valid range. Candidates are
//     reconstructed on-GPU from the stride residue table: the g-th valid
//     candidate at or after range start is B0 + (g/R)*M + residues[g%R],
//     so the host never transfers per-candidate data.
//   detailed_kernel: grid-stride over [start, start+count); each thread
//     derives its own n, so there is no input transfer at all. Unique-digit
//     counts accumulate in per-warp shared-memory histograms.

// Standalone fallbacks so the file can be syntax-checked without the host's
// define injection (values match base 40, k=2).
#if !defined(NICEONLY) && !defined(DETAILED)
#define NICEONLY
#define DETAILED
// PREFILTER may only default on in standalone syntax-check builds. The host
// omits it deliberately for bases whose n^2/n^3 are not guaranteed PRE_DIGITS
// digits: the low-digit peel would extract phantom leading zeros and falsely
// reject nice numbers (that bug shipped when this was an #ifndef fallback
// below, silently force-enabling the prefilter on bases 10-25).
#define PREFILTER
#define PRE_DIGITS 9
#define PRE_MOD 262144000000000ull
#define POW64_MOD_PRE 195081709551616ull
#endif
#ifndef BASE
#define BASE 40
#endif
#ifndef N_LIMBS
#define N_LIMBS 2
#endif
#ifndef CHUNK_DIGITS
#define CHUNK_DIGITS 5
#endif
#ifndef CHUNK_DIV
#define CHUNK_DIV 102400000u
#endif
#ifdef NICEONLY
#ifndef STRIDE_M
#define STRIDE_M 62400u
#endif
#ifndef STRIDE_R
#define STRIDE_R 4992u
#endif
#ifndef POW64_MOD_M
#define POW64_MOD_M 30016u
#endif
#endif
#ifdef DETAILED
#ifndef NEAR_MISS_CUTOFF
#define NEAR_MISS_CUTOFF 36
#endif
#endif

#if BASE > 128
#error "DigitSet is two u64 words; bases above 128 are not representable"
#endif

typedef unsigned long long u64;
typedef unsigned int u32;

// Number of threads per block. The host must launch detailed_kernel with
// exactly this block size (the shared-memory histogram is sized from it).
#define BLOCK_THREADS 256
#define WARPS_PER_BLOCK (BLOCK_THREADS / 32)

// Derived limb counts. n^2 needs at most 2*N_LIMBS limbs and n^3 at most
// 3*N_LIMBS. For base <= 68, n^3 < 2^256 so these never exceed 9 limbs.
#define SQ_LIMBS (2 * N_LIMBS)
#define CU_LIMBS (3 * N_LIMBS)

// ============================================================================
// Digit set: bitmask of seen digits. One u64 for bases <= 64, two above.
// ============================================================================

struct DigitSet {
    u64 lo;
#if BASE > 64
    u64 hi;
#endif
};

__device__ __forceinline__ void digitset_clear(DigitSet& s) {
    s.lo = 0;
#if BASE > 64
    s.hi = 0;
#endif
}

// Records digit d. If STOP_ON_DUP, returns false when d was already present.
template <bool STOP_ON_DUP>
__device__ __forceinline__ bool digitset_add(DigitSet& s, u32 d) {
#if BASE > 64
    u64& word = (d < 64) ? s.lo : s.hi;
    u64 bit = 1ull << (d & 63);
#else
    u64& word = s.lo;
    u64 bit = 1ull << d;
#endif
    if (STOP_ON_DUP && (word & bit) != 0) {
        return false;
    }
    word |= bit;
    return true;
}

// Records digit d and returns 1 if it was already present, 0 otherwise.
// Branchless, for the uniform-length prefilter.
__device__ __forceinline__ u32 digitset_test_and_set(DigitSet& s, u32 d) {
#if BASE > 64
    u64& word = (d < 64) ? s.lo : s.hi;
    u64 bit = 1ull << (d & 63);
#else
    u64& word = s.lo;
    u64 bit = 1ull << d;
#endif
    u32 dup = (word & bit) != 0 ? 1 : 0;
    word |= bit;
    return dup;
}

__device__ __forceinline__ u32 digitset_count(const DigitSet& s) {
#if BASE > 64
    return (u32)(__popcll(s.lo) + __popcll(s.hi));
#else
    return (u32)__popcll(s.lo);
#endif
}

// ============================================================================
// Multi-limb arithmetic (u32 limbs, little-endian)
// ============================================================================

// r[ra+rb limbs] = a[ra limbs] * b[rb limbs]. Schoolbook; r must not alias.
__device__ __forceinline__ void mul_limbs(
    const u32* a, int ra, const u32* b, int rb, u32* r
) {
    for (int i = 0; i < ra + rb; i++) {
        r[i] = 0;
    }
    for (int i = 0; i < ra; i++) {
        u64 carry = 0;
        for (int j = 0; j < rb; j++) {
            u64 cur = (u64)a[i] * b[j] + r[i + j] + carry;
            r[i + j] = (u32)cur;
            carry = cur >> 32;
        }
        r[i + rb] = (u32)carry;
    }
}

// Index of the highest nonzero limb, or -1 if the value is zero.
__device__ __forceinline__ int top_limb(const u32* v, int len) {
    int top = len - 1;
    while (top >= 0 && v[top] == 0) {
        top--;
    }
    return top;
}

// ============================================================================
// Digit extraction
// ============================================================================
//
// Chunked radix extraction: repeatedly split off v mod CHUNK_DIV (which holds
// CHUNK_DIGITS base-BASE digits), then peel single digits from the u32 chunk.
// Both divisions are by compile-time constants. Digit semantics match the
// CPU's `while n != 0 { d = n % b; n /= b; }` exactly: intermediate chunks
// contribute all CHUNK_DIGITS digits (including zeros), the most significant
// chunk contributes digits only until it reaches zero.

// Extracts every digit of v[0..=top] into `set`. If STOP_ON_DUP, returns
// false as soon as a duplicate digit is seen. Destroys v.
template <bool STOP_ON_DUP>
__device__ __forceinline__ bool scan_digits(u32* v, int top, DigitSet& set) {
    while (top >= 0) {
        // Split: rem = v mod CHUNK_DIV, v /= CHUNK_DIV.
        // rem < CHUNK_DIV < 2^31, so cur < 2^63.
        u32 rem = 0;
        for (int i = top; i >= 0; i--) {
            u64 cur = ((u64)rem << 32) | v[i];
            u64 q = cur / CHUNK_DIV; // const divisor -> multiply-high
            rem = (u32)(cur - q * CHUNK_DIV);
            v[i] = (u32)q;
        }
        while (top >= 0 && v[top] == 0) {
            top--;
        }

        u32 chunk = rem;
        if (top >= 0) {
            // Full interior chunk: exactly CHUNK_DIGITS digits, zeros included.
#pragma unroll
            for (int k = 0; k < CHUNK_DIGITS; k++) {
                u32 d = chunk % BASE; // const divisor -> multiply-high
                chunk /= BASE;
                if (!digitset_add<STOP_ON_DUP>(set, d)) {
                    return false;
                }
            }
        } else {
            // Most significant chunk: digits until zero.
            while (chunk != 0) {
                u32 d = chunk % BASE;
                chunk /= BASE;
                if (!digitset_add<STOP_ON_DUP>(set, d)) {
                    return false;
                }
            }
        }
    }
    return true;
}

// Unpacks n = (n_lo, n_hi) into N_LIMBS u32 limbs and computes n^2 and n^3.
__device__ __forceinline__ void square_and_cube(
    u64 n_lo, u64 n_hi, u32* sq, u32* cu
) {
    u32 n32[N_LIMBS];
#pragma unroll
    for (int i = 0; i < N_LIMBS; i++) {
        u64 word = (i < 2) ? n_lo : n_hi;
        n32[i] = (u32)(word >> ((i & 1) * 32));
    }
    mul_limbs(n32, N_LIMBS, n32, N_LIMBS, sq);
    mul_limbs(sq, SQ_LIMBS, n32, N_LIMBS, cu);
}

// Nice check with early exit on the first duplicate digit (niceonly mode).
__device__ __forceinline__ bool check_is_nice(u64 n_lo, u64 n_hi) {
    u32 sq[SQ_LIMBS];
    u32 cu[CU_LIMBS];
    square_and_cube(n_lo, n_hi, sq, cu);

    DigitSet set;
    digitset_clear(set);
    if (!scan_digits<true>(sq, top_limb(sq, SQ_LIMBS), set)) {
        return false;
    }
    if (!scan_digits<true>(cu, top_limb(cu, CU_LIMBS), set)) {
        return false;
    }
    return digitset_count(set) == BASE;
}

// Full unique-digit count, no early exit (detailed mode).
__device__ __forceinline__ u32 num_unique_digits(u64 n_lo, u64 n_hi) {
    u32 sq[SQ_LIMBS];
    u32 cu[CU_LIMBS];
    square_and_cube(n_lo, n_hi, sq, cu);

    DigitSet set;
    digitset_clear(set);
    scan_digits<false>(sq, top_limb(sq, SQ_LIMBS), set);
    scan_digits<false>(cu, top_limb(cu, CU_LIMBS), set);
    return digitset_count(set);
}

// ============================================================================
// Kernel: niceonly
// ============================================================================

#ifdef NICEONLY

// n mod STRIDE_M for n = (lo, hi). Both % below are by the constant M.
__device__ __forceinline__ u32 mod_m(u64 n_lo, u64 n_hi) {
    u32 hi_mod = (u32)(n_hi % STRIDE_M);
    u32 lo_mod = (u32)(n_lo % STRIDE_M);
    // hi_mod * POW64_MOD_M + lo_mod < M^2 + M <= 2^42, fits u64.
    u64 t = (u64)hi_mod * POW64_MOD_M + lo_mod;
    return (u32)(t % STRIDE_M);
}

#ifdef PREFILTER

// Reduce value = hi*2^64 + lo mod PRE_MOD. The loop maintains the invariant
// value === hi*2^64 + lo (mod PRE_MOD) and shrinks hi by ~16 bits per step
// (PRE_MOD <= 2^48), so it terminates in at most a few iterations for any
// input; callers here always pass hi < 2^32 (two steps).
__device__ __forceinline__ u64 reduce_pre(u64 hi, u64 lo) {
    while (hi != 0) {
        u64 p_lo = hi * POW64_MOD_PRE;
        u64 p_hi = __umul64hi(hi, POW64_MOD_PRE);
        lo += p_lo;
        hi = p_hi + (lo < p_lo ? 1 : 0);
    }
    return lo % PRE_MOD; // const divisor -> multiply-high
}

// (a * b) mod PRE_MOD for a, b < PRE_MOD.
__device__ __forceinline__ u64 mulmod_pre(u64 a, u64 b) {
    return reduce_pre(__umul64hi(a, b), a * b);
}

// Cheap uniform pre-check: the lowest PRE_DIGITS digits of n^2 and of n^3,
// computed entirely in u64 modular arithmetic via
// x^k mod b^p == (x mod b^p)^k mod b^p — no multi-limb work at all.
// Returns false if any digit repeats (the candidate cannot be nice).
// Fixed-length and branch-free so warps stay converged; survivors
// (a few percent to ~25% depending on base) fall through to the full check,
// which redoes these digits from scratch — the recompute is amortized to
// noise by the kill rate, and it keeps check_is_nice untouched.
//
// Soundness requires n^2 and n^3 to really have at least PRE_DIGITS digits
// each (otherwise the loop would extract phantom leading zeros); the host
// only defines PREFILTER when the base's range guarantees that.
__device__ __forceinline__ bool prefilter_low_digits(u64 n_lo, u64 n_hi) {
    u64 nm = reduce_pre(n_hi, n_lo);
    u64 sq = mulmod_pre(nm, nm);
    u64 cu = mulmod_pre(sq, nm);

    DigitSet set;
    digitset_clear(set);
    u32 dup = 0;
#pragma unroll
    for (int k = 0; k < PRE_DIGITS; k++) {
        u32 d = (u32)(sq % BASE); // const divisor -> multiply-high
        sq /= BASE;
        dup |= digitset_test_and_set(set, d);
    }
#pragma unroll
    for (int k = 0; k < PRE_DIGITS; k++) {
        u32 d = (u32)(cu % BASE);
        cu /= BASE;
        dup |= digitset_test_and_set(set, d);
    }
    return dup == 0;
}

#endif // PREFILTER

// Full candidate check, with the modular prefilter in front when enabled.
__device__ __forceinline__ bool candidate_is_nice(u64 n_lo, u64 n_hi) {
#ifdef PREFILTER
    if (!prefilter_low_digits(n_lo, n_hi)) {
        return false;
    }
#endif
    return check_is_nice(n_lo, n_hi);
}

// First index in residues[0..STRIDE_R) with residues[idx] >= m, or STRIDE_R.
__device__ __forceinline__ u32 lower_bound_residue(
    const u32* __restrict__ residues, u32 m
) {
    u32 lo = 0;
    u32 hi = STRIDE_R;
    while (lo < hi) {
        u32 mid = (lo + hi) >> 1;
        if (residues[mid] < m) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

// One warp per MSD-valid range. Each range is (field_start + offset,
// field_start + offset + len). Lanes stride through the range's valid
// candidates by global residue-sequence index; the g-th candidate at or
// after the range start is B0 + (g / R) * M + residues[g % R], where
// B0 = range_start - (range_start mod M) and g starts at the lower bound
// of (range_start mod M) in the residue table.
extern "C" __global__ void niceonly_ranges_kernel(
    u64 field_start_lo,
    u64 field_start_hi,
    const u64* __restrict__ range_offsets,
    const u32* __restrict__ range_lens,
    u32 num_ranges,
    const u32* __restrict__ residues,
    u64* __restrict__ nice_out, // (lo, hi) pairs
    u32* __restrict__ nice_count,
    u32 nice_capacity
) {
    u32 gtid = blockIdx.x * blockDim.x + threadIdx.x;
    u32 warp = gtid >> 5;
    u32 lane = gtid & 31;
    u32 nwarps = (gridDim.x * blockDim.x) >> 5;

    for (u32 r = warp; r < num_ranges; r += nwarps) {
        // range_start = field_start + offset
        u64 rs_lo = field_start_lo + range_offsets[r];
        u64 rs_hi = field_start_hi + (rs_lo < field_start_lo ? 1 : 0);
        // range_end = range_start + len
        u64 re_lo = rs_lo + range_lens[r];
        u64 re_hi = rs_hi + (re_lo < rs_lo ? 1 : 0);

        u32 m = mod_m(rs_lo, rs_hi);
        // B0 = range_start - m (m < M << 2^32)
        u64 b0_lo = rs_lo - m;
        u64 b0_hi = rs_hi - (rs_lo < (u64)m ? 1 : 0);

        // All lanes compute the same search (cheap, keeps the warp uniform).
        u32 idx0 = lower_bound_residue(residues, m);

        for (u32 g = idx0 + lane;; g += 32) {
            u32 cycle = g / STRIDE_R; // const divisor -> multiply-high
            u32 j = g - cycle * STRIDE_R;
            u64 add = (u64)cycle * STRIDE_M + residues[j];
            u64 n_lo = b0_lo + add;
            u64 n_hi = b0_hi + (n_lo < b0_lo ? 1 : 0);
            if (n_hi > re_hi || (n_hi == re_hi && n_lo >= re_lo)) {
                break;
            }
            if (candidate_is_nice(n_lo, n_hi)) {
                u32 pos = atomicAdd(nice_count, 1);
                if (pos < nice_capacity) {
                    nice_out[2 * (size_t)pos] = n_lo;
                    nice_out[2 * (size_t)pos + 1] = n_hi;
                }
            }
        }
    }
}

#endif // NICEONLY

// ============================================================================
// Kernel: detailed
// ============================================================================

#ifdef DETAILED

#define HIST_BINS (BASE + 1)

// Grid-stride over [start, start + count). Each thread derives n = start + idx
// (no input arrays). Unique-digit counts go into per-warp shared histograms
// (contention relief), flushed to the global u64 histogram once per block.
// Numbers above NEAR_MISS_CUTOFF are appended to the miss buffer.
extern "C" __global__ void detailed_kernel(
    u64 start_lo,
    u64 start_hi,
    u64 count,
    u64* __restrict__ histogram, // HIST_BINS entries, accumulated across launches
    u64* __restrict__ miss_out,  // (lo, hi) pairs
    u32* __restrict__ miss_uniques,
    u32* __restrict__ miss_count,
    u32 miss_capacity
) {
    __shared__ u32 hist_s[WARPS_PER_BLOCK][HIST_BINS];

    for (u32 i = threadIdx.x; i < WARPS_PER_BLOCK * HIST_BINS; i += blockDim.x) {
        hist_s[i / HIST_BINS][i % HIST_BINS] = 0;
    }
    __syncthreads();

    u32 warp_in_block = threadIdx.x >> 5;
    u64 stride = (u64)gridDim.x * blockDim.x;
    for (u64 idx = (u64)blockIdx.x * blockDim.x + threadIdx.x; idx < count;
         idx += stride) {
        u64 n_lo = start_lo + idx;
        u64 n_hi = start_hi + (n_lo < start_lo ? 1 : 0);
        u32 u = num_unique_digits(n_lo, n_hi);
        atomicAdd(&hist_s[warp_in_block][u], 1);
        if (u > NEAR_MISS_CUTOFF) {
            u32 pos = atomicAdd(miss_count, 1);
            if (pos < miss_capacity) {
                miss_out[2 * (size_t)pos] = n_lo;
                miss_out[2 * (size_t)pos + 1] = n_hi;
                miss_uniques[pos] = u;
            }
        }
    }

    __syncthreads();
    for (u32 bin = threadIdx.x; bin < HIST_BINS; bin += blockDim.x) {
        u64 sum = 0;
        for (int w = 0; w < WARPS_PER_BLOCK; w++) {
            sum += hist_s[w][bin];
        }
        if (sum != 0) {
            atomicAdd(&histogram[bin], sum);
        }
    }
}

#endif // DETAILED
