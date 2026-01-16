// CUDA kernels for nice number checking
// Implements GPU-accelerated versions of the hot loop operations

typedef unsigned long long uint64_t;
typedef unsigned int uint32_t;
typedef unsigned char uint8_t;

// ============================================================================
// CUDA Intrinsic Helpers for Fast Arithmetic
// ============================================================================

// Fast 64x64->128 multiply using CUDA intrinsics
__device__ __forceinline__ void mul64x64_128(uint64_t a, uint64_t b, uint64_t& lo, uint64_t& hi) {
    lo = a * b;
    hi = __umul64hi(a, b);
}

// Fast multiply-add: result = a * b + c, with carry
__device__ __forceinline__ void mad64_wide(uint64_t a, uint64_t b, uint64_t c, uint64_t& lo, uint64_t& hi) {
    lo = a * b + c;
    hi = __umul64hi(a, b) + (lo < c ? 1 : 0);
}

// ============================================================================
// u128 Structure and Operations
// ============================================================================

struct u128 {
    uint64_t lo;
    uint64_t hi;

    __device__ __forceinline__ u128() : lo(0), hi(0) {}
    __device__ __forceinline__ u128(uint64_t l, uint64_t h) : lo(l), hi(h) {}
    __device__ __forceinline__ explicit u128(uint64_t l) : lo(l), hi(0) {}
};

// Fast u128 addition with carry
__device__ __forceinline__ u128 add_u128(const u128& a, const u128& b) {
    u128 result;
    result.lo = a.lo + b.lo;
    result.hi = a.hi + b.hi + (result.lo < a.lo ? 1 : 0);
    return result;
}

// Fast u128 multiplication using intrinsics
__device__ __forceinline__ void mul_u128_u128(const u128& a, const u128& b, uint64_t result[4]) {
    // Compute a * b = (a.hi * 2^64 + a.lo) * (b.hi * 2^64 + b.lo)
    uint64_t lo_lo_lo, lo_lo_hi;
    mul64x64_128(a.lo, b.lo, lo_lo_lo, lo_lo_hi);

    uint64_t lo_hi_lo, lo_hi_hi;
    mul64x64_128(a.lo, b.hi, lo_hi_lo, lo_hi_hi);

    uint64_t hi_lo_lo, hi_lo_hi;
    mul64x64_128(a.hi, b.lo, hi_lo_lo, hi_lo_hi);

    uint64_t hi_hi_lo, hi_hi_hi;
    mul64x64_128(a.hi, b.hi, hi_hi_lo, hi_hi_hi);

    // Accumulate results
    result[0] = lo_lo_lo;

    // result[1] = lo_lo_hi + lo_hi_lo + hi_lo_lo
    uint64_t temp1 = lo_lo_hi + lo_hi_lo;
    uint64_t carry1 = (temp1 < lo_lo_hi) ? 1 : 0;
    result[1] = temp1 + hi_lo_lo;
    uint64_t carry2 = (result[1] < temp1) ? 1 : 0;
    uint64_t carry = carry1 + carry2;

    // result[2] = lo_hi_hi + hi_lo_hi + hi_hi_lo + carry
    uint64_t temp2 = lo_hi_hi + hi_lo_hi;
    uint64_t carry3 = (temp2 < lo_hi_hi) ? 1 : 0;
    uint64_t temp3 = temp2 + hi_hi_lo;
    uint64_t carry4 = (temp3 < temp2) ? 1 : 0;
    result[2] = temp3 + carry;
    uint64_t carry5 = (result[2] < temp3) ? 1 : 0;
    carry = carry3 + carry4 + carry5;

    result[3] = hi_hi_hi + carry;
}

// ============================================================================
// Extended Precision u256 Structure
// ============================================================================

struct u256 {
    uint64_t limbs[4]; // limbs[0] = lowest 64 bits

    __device__ __forceinline__ u256() {
        #pragma unroll
        for (int i = 0; i < 4; i++) limbs[i] = 0;
    }

    __device__ __forceinline__ bool is_zero() const {
        return (limbs[0] | limbs[1] | limbs[2] | limbs[3]) == 0;
    }
};

// Fast squaring: n^2 using optimized multiplication
// Uses algebraic expansion: (a + b)^2 = a^2 + 2ab + b^2
// where a = hi * 2^64 and b = lo
__device__ __forceinline__ u256 square_u128_fast(const u128& n) {
    u256 result;

    // n^2 = (hi * 2^64 + lo)^2 = hi^2 * 2^128 + 2*hi*lo * 2^64 + lo^2
    uint64_t lo_sq_lo, lo_sq_hi;
    mul64x64_128(n.lo, n.lo, lo_sq_lo, lo_sq_hi);

    uint64_t hi_sq_lo, hi_sq_hi;
    mul64x64_128(n.hi, n.hi, hi_sq_lo, hi_sq_hi);

    uint64_t cross_lo, cross_hi;
    mul64x64_128(n.lo, n.hi, cross_lo, cross_hi);

    // Double the cross term (2 * lo * hi)
    uint64_t cross2_lo = cross_lo << 1;
    uint64_t cross2_hi = (cross_hi << 1) | (cross_lo >> 63);
    uint64_t cross2_carry = cross_hi >> 63;

    // Assemble result
    result.limbs[0] = lo_sq_lo;
    result.limbs[1] = lo_sq_hi + cross2_lo;
    uint64_t carry = (result.limbs[1] < lo_sq_hi) ? 1 : 0;

    uint64_t temp = cross2_hi + hi_sq_lo;
    uint64_t carry2 = (temp < cross2_hi) ? 1 : 0;
    result.limbs[2] = temp + carry;
    uint64_t carry3 = (result.limbs[2] < temp) ? 1 : 0;
    carry = carry2 + carry3;

    result.limbs[3] = hi_sq_hi + cross2_carry + carry;

    return result;
}

// Fast cubing: compute n^3 by multiplying n * n^2
__device__ __forceinline__ u256 cube_u128_fast(const u128& n) {
    u256 n_sq = square_u128_fast(n);
    u256 result;

    // Multiply 256-bit n_sq by 128-bit n
    uint64_t carry = 0;

    // n_sq.limbs[0] * n.lo
    uint64_t prod_lo, prod_hi;
    mul64x64_128(n_sq.limbs[0], n.lo, prod_lo, prod_hi);
    result.limbs[0] = prod_lo;
    carry = prod_hi;

    // n_sq.limbs[1] * n.lo + n_sq.limbs[0] * n.hi + carry
    uint64_t temp_lo, temp_hi;
    mul64x64_128(n_sq.limbs[1], n.lo, temp_lo, temp_hi);
    uint64_t sum1 = temp_lo + carry;
    uint64_t carry1 = (sum1 < temp_lo) ? 1 : 0;
    carry = temp_hi + carry1;

    mul64x64_128(n_sq.limbs[0], n.hi, temp_lo, temp_hi);
    result.limbs[1] = sum1 + temp_lo;
    uint64_t carry2 = (result.limbs[1] < sum1) ? 1 : 0;
    carry += temp_hi + carry2;

    // n_sq.limbs[2] * n.lo + n_sq.limbs[1] * n.hi + carry
    mul64x64_128(n_sq.limbs[2], n.lo, temp_lo, temp_hi);
    uint64_t sum2 = temp_lo + carry;
    carry1 = (sum2 < temp_lo) ? 1 : 0;
    carry = temp_hi + carry1;

    mul64x64_128(n_sq.limbs[1], n.hi, temp_lo, temp_hi);
    result.limbs[2] = sum2 + temp_lo;
    carry2 = (result.limbs[2] < sum2) ? 1 : 0;
    carry += temp_hi + carry2;

    // n_sq.limbs[3] * n.lo + n_sq.limbs[2] * n.hi + carry
    mul64x64_128(n_sq.limbs[3], n.lo, temp_lo, temp_hi);
    uint64_t sum3 = temp_lo + carry;
    // Ignore overflow beyond 256 bits

    mul64x64_128(n_sq.limbs[2], n.hi, temp_lo, temp_hi);
    result.limbs[3] = sum3 + temp_lo;

    return result;
}

// ============================================================================
// Division by Base
// ============================================================================

// Generic division for u256 by small base
__device__ __forceinline__ uint32_t div_u256_by_base_generic(u256& n, uint32_t base) {
    uint64_t remainder = 0;

    // Process from most significant to least significant
    #pragma unroll
    for (int i = 3; i >= 0; i--) {
        // Use 128-bit division emulation
        uint32_t limb_hi = (uint32_t)(n.limbs[i] >> 32);
        uint32_t limb_lo = (uint32_t)(n.limbs[i] & 0xFFFFFFFFULL);

        uint64_t dividend_hi = (remainder << 32) | limb_hi;
        uint64_t quotient_hi = dividend_hi / base;
        remainder = dividend_hi % base;

        uint64_t dividend_lo = (remainder << 32) | limb_lo;
        uint64_t quotient_lo = dividend_lo / base;
        remainder = dividend_lo % base;

        n.limbs[i] = (quotient_hi << 32) | quotient_lo;
    }

    return (uint32_t)remainder;
}

// Division dispatcher (currently just uses generic, can add specialized versions for common bases)
__device__ __forceinline__ uint32_t div_u256_by_base(u256& n, uint32_t base) {
    return div_u256_by_base_generic(n, base);
}

// ============================================================================
// Kernel 1: Count unique digits (detailed mode)
// ============================================================================

extern "C" __global__ void count_unique_digits_kernel(
    const uint64_t* numbers_lo,
    const uint64_t* numbers_hi,
    uint32_t* unique_counts,
    const uint32_t base,
    const size_t n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    u128 num(numbers_lo[idx], numbers_hi[idx]);

    // Calculate n^2 and n^3 using optimized functions
    u256 squared = square_u128_fast(num);
    u256 cubed = cube_u128_fast(num);

    // Track unique digits using bitmask (supports bases up to 128)
    uint64_t digits_lo = 0;
    uint64_t digits_hi = 0;

    // Process squared - extract digits in base
    u256 temp_sq = squared;
    while (!temp_sq.is_zero()) {
        uint32_t digit = div_u256_by_base(temp_sq, base);
        if (digit < 64) {
            digits_lo |= (1ULL << digit);
        } else {
            digits_hi |= (1ULL << (digit - 64));
        }
    }

    // Process cubed - extract digits in base
    u256 temp_cb = cubed;
    while (!temp_cb.is_zero()) {
        uint32_t digit = div_u256_by_base(temp_cb, base);
        if (digit < 64) {
            digits_lo |= (1ULL << digit);
        } else {
            digits_hi |= (1ULL << (digit - 64));
        }
    }

    // Count set bits
    unique_counts[idx] = __popcll(digits_lo) + __popcll(digits_hi);
}

// ============================================================================
// Kernel 2: Check if nice (niceonly mode)
// ============================================================================

extern "C" __global__ void check_is_nice_kernel(
    const uint64_t* numbers_lo,
    const uint64_t* numbers_hi,
    uint8_t* is_nice,
    const uint32_t base,
    const size_t n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    u128 num(numbers_lo[idx], numbers_hi[idx]);

    // Calculate n^2 and n^3 using optimized functions
    u256 squared = square_u128_fast(num);
    u256 cubed = cube_u128_fast(num);

    // Track unique digits with early exit on duplicates
    uint64_t digits_lo = 0;
    uint64_t digits_hi = 0;

    // Process squared with duplicate detection
    u256 temp_sq = squared;
    while (!temp_sq.is_zero()) {
        uint32_t digit = div_u256_by_base(temp_sq, base);

        // Check for duplicate and early exit
        if (digit < 64) {
            uint64_t mask = (1ULL << digit);
            if (digits_lo & mask) {
                is_nice[idx] = 0;
                return;
            }
            digits_lo |= mask;
        } else {
            uint64_t mask = (1ULL << (digit - 64));
            if (digits_hi & mask) {
                is_nice[idx] = 0;
                return;
            }
            digits_hi |= mask;
        }
    }

    // Process cubed with duplicate detection
    u256 temp_cb = cubed;
    while (!temp_cb.is_zero()) {
        uint32_t digit = div_u256_by_base(temp_cb, base);

        // Check for duplicate and early exit
        if (digit < 64) {
            uint64_t mask = (1ULL << digit);
            if (digits_lo & mask) {
                is_nice[idx] = 0;
                return;
            }
            digits_lo |= mask;
        } else {
            uint64_t mask = (1ULL << (digit - 64));
            if (digits_hi & mask) {
                is_nice[idx] = 0;
                return;
            }
            digits_hi |= mask;
        }
    }

    // Check if we have exactly 'base' unique digits
    uint32_t count = __popcll(digits_lo) + __popcll(digits_hi);
    is_nice[idx] = (count == base) ? 1 : 0;
}

// ============================================================================
// Kernel 3: Residue filter (preprocessing)
// ============================================================================

extern "C" __global__ void filter_by_residue_kernel(
    const uint64_t* numbers_lo,
    const uint64_t* numbers_hi,
    const uint64_t* filter_residues,
    const size_t filter_size,
    uint8_t* matches,
    const uint32_t base,
    const size_t n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    uint64_t num_lo = numbers_lo[idx];
    uint64_t num_hi = numbers_hi[idx];
    uint64_t base_minus_one = base - 1;

    // Compute u128 modulo (base-1): (hi * 2^64 + lo) % (base-1)
    uint64_t hi_mod = num_hi % base_minus_one;
    uint64_t lo_mod = num_lo % base_minus_one;

    // Compute 2^64 % (base-1)
    uint64_t power_mod = ((1ULL << 63) % base_minus_one);
    power_mod = (power_mod + power_mod) % base_minus_one;

    uint64_t residue = (hi_mod * power_mod + lo_mod) % base_minus_one;

    // Linear search through filter (could use binary search for large filters)
    matches[idx] = 0;
    #pragma unroll 4
    for (size_t i = 0; i < filter_size; i++) {
        if (residue == filter_residues[i]) {
            matches[idx] = 1;
            return;
        }
    }
}
