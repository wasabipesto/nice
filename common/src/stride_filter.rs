//! Stride-based iteration using the Chinese Remainder Theorem (CRT).
//!
//! Instead of iterating through every integer and filtering, we use CRT to combine
//! the residue filter (mod b-1) and the multi-digit LSD filter (mod b^k) into a single
//! modulus M = (b-1) × b^k.
//!
//! We precompute which residues mod M are valid, then iterate by jumping directly from
//! one valid candidate to the next using a gap table. This has zero filter overhead
//! per candidate - we simply never visit invalid candidates.

use crate::client_process::get_is_nice_with_known_lsd;
use crate::{FieldSize, NiceNumberSimple, affine_filter, lsd_filter, residue_filter};
use log::trace;

/// A precomputed stride table for efficient CRT-based iteration.
///
/// This table combines the residue filter (mod b-1) and multi-digit LSD filter (mod b^k)
/// into a single modulus using the Chinese Remainder Theorem. Instead of checking filters
/// for each candidate, we can jump directly from one valid candidate to the next.
///
/// Residues and gaps are stored as `u32`. This caps the modulus at `u32::MAX`,
/// which any base ≤ 256 with k ≤ 3 satisfies, and keeps the table compact:
/// at k=3 the residue count reaches several hundred thousand entries, where
/// 32-byte-per-entry storage would blow past cache while 8 bytes stays cheap
/// (entry lookups binary-search the residues; iteration streams only gaps).
pub struct StrideTable {
    /// The combined modulus: M = (b-1) × b^k
    pub modulus: u128,
    /// The number of low digits fixed by each residue (the LSD filter depth)
    pub k: u32,
    /// Sorted list of valid residues mod M
    pub valid_residues: Vec<u32>,
    /// Gap from each valid residue to the next: `gap_table[i] = valid_residues[i+1] - valid_residues[i]`
    /// The last entry wraps around: `gap_table[last] = M - valid_residues[last] + valid_residues[0]`
    pub gap_table: Vec<u32>,
    /// Per-residue bitmask of the 2k fixed low digits of n² and n³ (bit d set
    /// = digit d appears). All 2k digits are pairwise distinct by
    /// construction (the LSD filter rejected everything else), so the nice
    /// check can seed its duplicate indicator from this mask and skip
    /// re-extracting the low digits. Empty when base > 64 (digits would not
    /// Lean: `Nice.lowMask_eq` (STR-4)
    /// Lean: `Nice.Const.mask_width` (NUM-5)
    /// fit a u64 mask); iteration then falls back to the unseeded check.
    pub low_digit_masks: Vec<u64>,
}

impl StrideTable {
    /// Create a new stride table for the given base and k-digit LSD filter.
    ///
    /// # Arguments
    /// - `base`: The numeric base
    /// - `k`: Number of least significant digits to check (from multi-digit LSD filter)
    ///
    /// Lean: `Nice.mem_validResidues_iff` (STR-1)
    /// Lean: `Nice.Const.stride_modulus_u32` (NUM-4)
    ///
    /// # Panics
    /// Panics if base^k overflows u32 or (base-1) × base^k overflows u32
    #[must_use]
    pub fn new(base: u32, k: u32) -> Self {
        let b_minus_1 = base - 1;
        let b_k = base.checked_pow(k).expect("base^k must fit in u32");
        let modulus = b_minus_1
            .checked_mul(b_k)
            .expect("(base-1) * base^k must fit in u32"); // CRT: gcd(b-1, b^k) = 1

        // Get the residue filter valid set (mod b-1) as a direct-index table
        let residue_set = residue_filter::get_residue_filter(&base);
        let mut residue_ok = vec![false; b_minus_1 as usize];
        for r in residue_set {
            residue_ok[r as usize] = true;
        }

        // Get the multi-digit LSD filter bitmap (mod b^k)
        let lsd_bitmap = lsd_filter::get_valid_multi_lsd_bitmap(base, k);

        // Find all residues r mod M that satisfy both filters
        let mut valid_residues = Vec::new();
        for r in 0..modulus {
            let passes_residue = residue_ok[(r % b_minus_1) as usize];
            let passes_lsd = lsd_bitmap[(r % b_k) as usize];
            if passes_residue && passes_lsd {
                valid_residues.push(r);
            }
        }

        // Compute gaps between consecutive valid residues
        let mut gap_table = Vec::with_capacity(valid_residues.len());
        for i in 0..valid_residues.len() {
            let next_gap = if i + 1 < valid_residues.len() {
                valid_residues[i + 1] - valid_residues[i]
            } else {
                // Wraparound: distance from last valid residue back to first
                modulus - valid_residues[i] + valid_residues[0]
            };
            gap_table.push(next_gap);
        }

        // Precompute the fixed low digits of n² and n³ for each residue so
        // the nice check can skip re-extracting them (see `low_digit_masks`).
        let low_digit_masks = if base <= 64 {
            let b_k_u64 = u64::from(b_k);
            valid_residues
                .iter()
                .map(|&r| {
                    let suffix = u64::from(r % b_k);
                    let mut mask: u64 = 0;
                    for value in [
                        suffix * suffix % b_k_u64,
                        suffix * suffix % b_k_u64 * suffix % b_k_u64,
                    ] {
                        let mut v = value;
                        for _ in 0..k {
                            mask |= 1 << (v % u64::from(base));
                            v /= u64::from(base);
                        }
                    }
                    mask
                })
                .collect()
        } else {
            Vec::new()
        };

        #[allow(clippy::cast_precision_loss)]
        {
            trace!(
                "Stride table for base {base} k={k}: modulus={modulus}, {} valid residues ({:.2}% pass rate)",
                valid_residues.len(),
                100.0 * valid_residues.len() as f64 / f64::from(modulus)
            );
        }

        StrideTable {
            modulus: u128::from(modulus),
            k,
            valid_residues,
            gap_table,
            low_digit_masks,
        }
    }

    /// Find the first valid candidate >= start and return `(candidate, gap_index)`.
    ///
    /// # Arguments
    /// - `start`: The starting value
    ///
    /// # Returns
    /// A tuple of `(first_valid_n, gap_index)` where:
    /// - `first_valid_n` is the smallest n >= start with n % M in `valid_residues`
    /// - `gap_index` is the index in `valid_residues`/`gap_table` for this residue
    #[must_use]
    pub fn first_valid_at_or_after(&self, start: u128) -> (u128, usize) {
        // The modulus fits in u32 (enforced at construction), so the residue does too.
        #[allow(clippy::cast_possible_truncation)]
        let r = (start % self.modulus) as u32;

        // Binary search for the first valid residue >= r
        let idx = match self.valid_residues.binary_search(&r) {
            Ok(i) => i, // Exact match
            Err(i) => {
                if i < self.valid_residues.len() {
                    i // First residue > r
                } else {
                    0 // Wrapped around, use first residue
                }
            }
        };

        let target_r = u128::from(self.valid_residues[idx]);
        let r = u128::from(r);
        let n = if target_r >= r {
            // Same cycle: just advance to target_r
            start + (target_r - r)
        } else {
            // Next cycle: wrap around the modulus
            start + (self.modulus - r + target_r)
        };

        (n, idx)
    }

    /// Iterate over all valid candidates in the range, applying `get_is_nice` to each.
    ///
    /// This is the core stride-based iteration function. Instead of checking every
    /// integer in the range, we jump directly from one valid candidate to the next
    /// using the precomputed gap table.
    ///
    /// # Arguments
    /// - `range`: The range to process
    /// - `base`: The numeric base
    ///
    /// # Returns
    /// A vector of nice numbers found in the range
    #[must_use]
    pub fn iterate_range(&self, range: &FieldSize, base: u32) -> Vec<NiceNumberSimple> {
        self.iterate_range_masked(range, base, 0)
    }

    /// [`StrideTable::iterate_range`] with the cross-end residue filter:
    /// `high_mask` holds digits the MSD analysis proved occupy some output
    /// position `>= k` for every `n` in this range
    /// (`msd_prefix_filter::MsdAnalysis`). A residue whose exact low-digit
    /// mask intersects it would repeat a digit across two distinct
    /// positions, so its candidates are skipped without a nice check —
    /// one AND on a mask this loop already loads.
    ///
    /// Survivors of that test then go through the affine middle-digit
    /// filter ([`affine_filter`]): the next `k` digits of each power
    /// (positions `k..2k`), computed from `n mod b^{2k}` with word
    /// arithmetic, are tested against the union of both masks before the
    /// full check runs.
    ///
    /// `high_mask` must be the certificate of an analysed range that
    /// contains `range` and has at least two numbers (what
    /// `msd_prefix_filter`'s recursion hands out: it only analyses ranges
    /// larger than its floor), with positions below `k` excluded (which
    /// `analyze_range(_, _, k)` guarantees); pass 0 to disable. The affine
    /// filter additionally needs the certificate to hold no digit from
    /// positions `k..2k`. That holds whenever the range starts at or above
    /// `b^{2k-1}`: two consecutive squares there already differ by more
    /// than `b^{2k-1}`, so none of those positions can be the same across
    /// the analysed range. The filter is only enabled for such ranges, which
    /// covers every legal range of every supported base (`b^5` is at most
    /// 1.1e9; legal ranges start above 1e12).
    ///
    /// Lean: `Nice.walk_eq_filter` (STR-2) — the walk visits exactly the valid
    /// candidates of the range, in order.
    #[must_use]
    pub fn iterate_range_masked(
        &self,
        range: &FieldSize,
        base: u32,
        high_mask: u64,
    ) -> Vec<NiceNumberSimple> {
        let use_affine = affine_filter::supports(base, self.k)
            && range.start() >= u128::from(base).pow(2 * self.k - 1);
        self.iterate_range_impl(range, base, high_mask, use_affine)
    }

    /// [`StrideTable::iterate_range_masked`] without the affine middle-digit
    /// filter (the stride table and the cross-end test still apply): the
    /// reference path for parity tests and A/B measurements.
    #[must_use]
    pub fn iterate_range_masked_without_affine(
        &self,
        range: &FieldSize,
        base: u32,
        high_mask: u64,
    ) -> Vec<NiceNumberSimple> {
        self.iterate_range_impl(range, base, high_mask, false)
    }

    #[inline]
    fn iterate_range_impl(
        &self,
        range: &FieldSize,
        base: u32,
        high_mask: u64,
        use_affine: bool,
    ) -> Vec<NiceNumberSimple> {
        self.walk_masked(range, base, high_mask, use_affine, |n, low| {
            get_is_nice_with_known_lsd(n, base, self.k, low)
        })
    }

    /// The masked walk with the seeded nice check injected: `check(n, low)`
    /// runs on exactly the candidates that pass the cross-end test and, with
    /// `use_affine`, the affine filter. Production passes the seeded nice
    /// check; tests pass a recorder to see which candidates got through.
    /// With the affine filter on (specialized bases) this is the two-phase
    /// walk; everything else takes the plain loop below.
    #[inline]
    fn walk_masked(
        &self,
        range: &FieldSize,
        base: u32,
        high_mask: u64,
        use_affine: bool,
        mut check: impl FnMut(u128, u64) -> bool,
    ) -> Vec<NiceNumberSimple> {
        if use_affine && !self.low_digit_masks.is_empty() && range.size() < (1u128 << 62) {
            return self.walk_two_phase(range, base, high_mask, check);
        }

        let mut results = Vec::new();
        let (mut n, mut idx) = self.first_valid_at_or_after(range.start());

        // Seed the nice check with each residue's known low digits when
        // masks are available (base ≤ 64). `get_is_nice_with_known_lsd`
        // itself falls back to the plain check for unspecialized bases.
        // (For bases above 64 the mask table is empty and `high_mask` is
        // always 0 — the analysis never emits mask bits there.)
        let masks = &self.low_digit_masks;

        while n < range.end() {
            let is_nice = if masks.is_empty() {
                crate::client_process::get_is_nice(n, base)
            } else {
                let low = masks[idx];
                low & high_mask == 0 && check(n, low)
            };
            if is_nice {
                results.push(NiceNumberSimple {
                    number: n,
                    num_uniques: base,
                });
            }
            n += u128::from(self.gap_table[idx]);
            idx += 1;
            if idx == self.gap_table.len() {
                idx = 0;
            }
        }

        results
    }

    /// Two-phase walk for specialized bases.
    ///
    /// Phase 1 streams the gap and low-mask tables in blocks, keeping each
    /// candidate's offset from the range start, `n mod b^{2k}` (for the
    /// affine filter; every gap is at most `(b-1)·b^k < b^{2k}`, so one
    /// conditional subtraction keeps it reduced) and low mask, and compacts
    /// the cross-end survivors (`low & high_mask == 0`) into small buffers
    /// with no data-dependent branch: 8 candidates per step under AVX-512, 4
    /// under AVX2, or an unconditional-store/conditional-increment scalar
    /// loop elsewhere. Phase 2 runs the affine filter and `check` on the
    /// survivors only. Same candidates, same checks, same results as the
    /// plain loop plus the affine filter.
    fn walk_two_phase(
        &self,
        range: &FieldSize,
        base: u32,
        high_mask: u64,
        mut check: impl FnMut(u128, u64) -> bool,
    ) -> Vec<NiceNumberSimple> {
        let k = self.k;
        let bk = u64::from(base).pow(k);
        let b2k = bk * bk;
        let start = range.start();
        let (n0, mut idx) = self.first_valid_at_or_after(start);
        // `range.size() < 2^62` and `n0 < start + M`, so offsets fit u64.
        #[allow(clippy::cast_possible_truncation)]
        let end_off = (range.end() - start) as u64;
        #[allow(clippy::cast_possible_truncation)]
        let mut off = (n0 - start) as u64;
        #[allow(clippy::cast_possible_truncation)]
        let mut nmod = (n0 % u128::from(b2k)) as u64;

        let mut results = Vec::new();
        SURVIVOR_BUFS.with(|cell| {
            let mut guard = cell.borrow_mut();
            let bufs: &mut SurvivorBufs = &mut guard;
            while off < end_off {
                let cursor = Cursor { idx, off, nmod };
                let (cnt, next) = phase1(
                    &self.gap_table,
                    &self.low_digit_masks,
                    cursor,
                    end_off,
                    b2k,
                    high_mask,
                    bufs,
                );
                for i in 0..cnt {
                    let low = bufs.lows[i];
                    if affine_filter::survives(base, bufs.nmods[i], low | high_mask) {
                        let n = start + u128::from(bufs.offs[i]);
                        if check(n, low) {
                            results.push(NiceNumberSimple {
                                number: n,
                                num_uniques: base,
                            });
                        }
                    }
                }
                idx = next.idx;
                off = next.off;
                nmod = next.nmod;
            }
        });
        results
    }

    /// Cross-end survivors a masked walk of `range` hands to phase 2, for
    /// tests comparing the block-wise phase 1 against a plain count.
    #[cfg(test)]
    #[allow(clippy::cast_possible_truncation)]
    fn two_phase_survivor_count(&self, range: &FieldSize, base: u32, high_mask: u64) -> usize {
        let bk = u64::from(base).pow(self.k);
        let b2k = bk * bk;
        let start = range.start();
        let (n0, idx) = self.first_valid_at_or_after(start);
        let end_off = (range.end() - start) as u64;
        let mut cursor = Cursor {
            idx,
            off: (n0 - start) as u64,
            nmod: (n0 % u128::from(b2k)) as u64,
        };
        let mut bufs = SurvivorBufs::new();
        let mut total = 0;
        while cursor.off < end_off {
            let (cnt, next) = phase1(
                &self.gap_table,
                &self.low_digit_masks,
                cursor,
                end_off,
                b2k,
                high_mask,
                &mut bufs,
            );
            total += cnt;
            cursor = next;
        }
        total
    }
}

/// Candidates per phase-1 block; bounds the survivor buffers.
const BLOCK: usize = 1024;

/// Phase-1 output: survivors' offsets from the range start, `n mod b^{2k}`
/// and low masks, structure-of-arrays. Sized for a full block plus the
/// unmasked 8-lane store past the last survivor.
struct SurvivorBufs {
    offs: [u64; BLOCK + 16],
    nmods: [u64; BLOCK + 16],
    lows: [u64; BLOCK + 16],
}

impl SurvivorBufs {
    fn new() -> Self {
        Self {
            offs: [0; BLOCK + 16],
            nmods: [0; BLOCK + 16],
            lows: [0; BLOCK + 16],
        }
    }
}

thread_local! {
    /// The buffers, once per thread: they are 25 KB, and a leaf at the MSD
    /// floor holds a few hundred candidates, so zeroing a fresh set per leaf
    /// would cost as much as walking it.
    static SURVIVOR_BUFS: std::cell::RefCell<Box<SurvivorBufs>> =
        std::cell::RefCell::new(Box::new(SurvivorBufs::new()));
}

/// Position in the walk: residue index, offset from the range start, and
/// `n mod b^{2k}` at that candidate.
#[derive(Clone, Copy)]
struct Cursor {
    idx: usize,
    off: u64,
    nmod: u64,
}

/// Phase-1 implementation selected for this process.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Phase1Impl {
    Scalar,
    #[cfg(target_arch = "x86_64")]
    Avx2,
    #[cfg(target_arch = "x86_64")]
    Avx512,
}

/// The best implementation this CPU supports, detected once and logged.
/// `NICE_SIMD` caps it for A/B testing: `0`, `off` or `scalar` forces the
/// scalar loop, `avx2` the AVX2 one, `avx512` (or unset) takes the best
/// available; anything else is reported and ignored.
fn phase1_impl() -> Phase1Impl {
    static IMPL: std::sync::OnceLock<Phase1Impl> = std::sync::OnceLock::new();
    *IMPL.get_or_init(|| {
        let cap = std::env::var("NICE_SIMD")
            .unwrap_or_default()
            .trim()
            .to_ascii_lowercase();
        let (scalar_only, no_avx512) = match cap.as_str() {
            "" | "avx512" => (false, false),
            "0" | "off" | "scalar" => (true, true),
            "avx2" => (false, true),
            other => {
                log::warn!("ignoring unknown NICE_SIMD '{other}' (0, avx2 or avx512)");
                (false, false)
            }
        };
        let chosen = if scalar_only {
            Phase1Impl::Scalar
        } else {
            best_phase1_impl(no_avx512)
        };
        log::debug!("stride walk phase 1: {chosen:?}");
        chosen
    })
}

/// The fastest phase 1 this CPU can run, AVX-512 excluded if asked.
#[cfg_attr(not(target_arch = "x86_64"), allow(unused_variables))]
fn best_phase1_impl(no_avx512: bool) -> Phase1Impl {
    #[cfg(target_arch = "x86_64")]
    {
        let avx2 = std::arch::is_x86_feature_detected!("avx2");
        // The AVX-512 function is also compiled with AVX2 enabled (rustc's
        // avx512f implies it), so require both: a VM can expose one without
        // the other.
        if !no_avx512 && avx2 && std::arch::is_x86_feature_detected!("avx512f") {
            return Phase1Impl::Avx512;
        }
        if avx2 {
            return Phase1Impl::Avx2;
        }
    }
    Phase1Impl::Scalar
}

/// The phase-1 implementation this process runs, for telemetry:
/// `"avx512"`, `"avx2"` or `"scalar"`.
#[must_use]
pub fn simd_tier() -> &'static str {
    match phase1_impl() {
        Phase1Impl::Scalar => "scalar",
        #[cfg(target_arch = "x86_64")]
        Phase1Impl::Avx2 => "avx2",
        #[cfg(target_arch = "x86_64")]
        Phase1Impl::Avx512 => "avx512",
    }
}

/// Run phase 1 over at most one block of candidates from `cursor`, writing
/// survivors to `bufs`. Returns the survivor count and the cursor to resume
/// from (`off >= end_off` when the range is exhausted).
#[inline]
fn phase1(
    gaps: &[u32],
    masks: &[u64],
    cursor: Cursor,
    end_off: u64,
    b2k: u64,
    high_mask: u64,
    bufs: &mut SurvivorBufs,
) -> (usize, Cursor) {
    // The SIMD loads are bounded by `gaps.len()` alone, and the AVX2 path's
    // signed offset compares need `end_off < 2^62` (the caller's bound).
    assert_eq!(gaps.len(), masks.len());
    debug_assert!(end_off < 1 << 62);
    match phase1_impl() {
        Phase1Impl::Scalar => {
            phase1_scalar(gaps, masks, cursor, end_off, b2k, high_mask, bufs, 0, BLOCK)
        }
        // SAFETY: the features were detected on this CPU.
        #[cfg(target_arch = "x86_64")]
        Phase1Impl::Avx2 => unsafe {
            phase1_avx2(gaps, masks, cursor, end_off, b2k, high_mask, bufs)
        },
        #[cfg(target_arch = "x86_64")]
        Phase1Impl::Avx512 => unsafe {
            phase1_avx512(gaps, masks, cursor, end_off, b2k, high_mask, bufs)
        },
    }
}

/// Scalar phase 1: unconditional store, conditional increment. Writes
/// survivors from buffer index `cnt` on, walks at most `max_candidates`
/// candidates, and returns the new survivor count.
#[inline]
#[allow(clippy::too_many_arguments)]
fn phase1_scalar(
    gaps: &[u32],
    masks: &[u64],
    mut cur: Cursor,
    end_off: u64,
    b2k: u64,
    high_mask: u64,
    bufs: &mut SurvivorBufs,
    mut cnt: usize,
    max_candidates: usize,
) -> (usize, Cursor) {
    let glen = gaps.len();
    let mut seen = 0usize;
    while cur.off < end_off && seen < max_candidates {
        let low = masks[cur.idx];
        bufs.offs[cnt] = cur.off;
        bufs.nmods[cnt] = cur.nmod;
        bufs.lows[cnt] = low;
        cnt += usize::from(low & high_mask == 0);
        let gap = u64::from(gaps[cur.idx]);
        cur.off += gap;
        cur.nmod += gap;
        if cur.nmod >= b2k {
            cur.nmod -= b2k;
        }
        cur.idx += 1;
        if cur.idx == glen {
            cur.idx = 0;
        }
        seen += 1;
    }
    (cnt, cur)
}

/// Lane-compaction table for the AVX2 path: for each 4-bit keep mask, the
/// `vpermd` index vector that moves the kept 64-bit lanes (as i32 pairs) to
/// the front.
#[cfg(target_arch = "x86_64")]
static AVX2_COMPRESS: [[i32; 8]; 16] = {
    let mut table = [[0i32; 8]; 16];
    let mut mask = 0usize;
    while mask < 16 {
        let mut dst = 0usize;
        let mut lane = 0i32;
        while lane < 4 {
            if mask >> lane & 1 == 1 {
                table[mask][2 * dst] = 2 * lane;
                table[mask][2 * dst + 1] = 2 * lane + 1;
                dst += 1;
            }
            lane += 1;
        }
        mask += 1;
    }
    table
};

/// AVX2 phase 1: four candidates per step, the same scheme as the AVX-512
/// routine with `vpermq` for the lane shifts, signed 64-bit compares (every
/// operand is far below 2^63) and a table-driven `vpermd` for the
/// compaction.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation,
    clippy::cast_ptr_alignment, // the loads are the unaligned forms
    clippy::too_many_lines
)]
unsafe fn phase1_avx2(
    gaps: &[u32],
    masks: &[u64],
    mut cur: Cursor,
    end_off: u64,
    b2k: u64,
    high_mask: u64,
    bufs: &mut SurvivorBufs,
) -> (usize, Cursor) {
    use std::arch::x86_64::{
        __m128i, __m256i, _mm_loadu_si128, _mm256_add_epi64, _mm256_and_si256, _mm256_castsi256_pd,
        _mm256_cmpeq_epi64, _mm256_cmpgt_epi64, _mm256_cvtepu32_epi64, _mm256_extract_epi64,
        _mm256_loadu_si256, _mm256_movemask_pd, _mm256_permute4x64_epi64,
        _mm256_permutevar8x32_epi32, _mm256_set_epi64x, _mm256_set1_epi64x, _mm256_setzero_si256,
        _mm256_storeu_si256, _mm256_sub_epi64,
    };
    let glen = gaps.len();
    let mut cnt = 0usize;
    let mut seen = 0usize;
    let vhi = _mm256_set1_epi64x(high_mask as i64);
    let vb2k = _mm256_set1_epi64x(b2k as i64);
    let vb2k_m1 = _mm256_set1_epi64x(b2k as i64 - 1);
    let vend = _mm256_set1_epi64x(end_off as i64);
    let zero = _mm256_setzero_si256();
    // Keep lanes 1..3 / 2..3 after a lane shift (set_epi64x lists lane 3 first).
    let keep_upper3 = _mm256_set_epi64x(-1, -1, -1, 0);
    let keep_upper2 = _mm256_set_epi64x(-1, -1, 0, 0);
    while cur.off < end_off && seen < BLOCK {
        if cur.idx + 4 > glen {
            // Tail of the residue table before the wrap: scalar.
            let n = glen - cur.idx;
            let (c, next) = phase1_scalar(gaps, masks, cur, end_off, b2k, high_mask, bufs, cnt, n);
            cnt = c;
            cur = next;
            seen += n;
            continue;
        }
        // SAFETY: cur.idx + 4 <= glen == masks.len(); loads stay in bounds.
        let (g32, vlow) = unsafe {
            (
                _mm_loadu_si128(gaps.as_ptr().add(cur.idx).cast::<__m128i>()),
                _mm256_loadu_si256(masks.as_ptr().add(cur.idx).cast::<__m256i>()),
            )
        };
        let mut g = _mm256_cvtepu32_epi64(g32);
        // Inclusive prefix sum: lanes [a, b, c, d] -> [a, a+b, a+b+c, a+b+c+d].
        // permute4x64 imm 0b10_01_00_00 = lanes (0, 0, 1, 2).
        g = _mm256_add_epi64(
            g,
            _mm256_and_si256(_mm256_permute4x64_epi64::<0b10_01_00_00>(g), keep_upper3),
        );
        // imm 0b01_00_00_00 = lanes (0, 0, 0, 1).
        g = _mm256_add_epi64(
            g,
            _mm256_and_si256(_mm256_permute4x64_epi64::<0b01_00_00_00>(g), keep_upper2),
        );
        // Exclusive prefix: lane i holds the sum of gaps before candidate i.
        let ex = _mm256_and_si256(_mm256_permute4x64_epi64::<0b10_01_00_00>(g), keep_upper3);
        let voff = _mm256_add_epi64(_mm256_set1_epi64x(cur.off as i64), ex);
        let mut vnmod = _mm256_add_epi64(_mm256_set1_epi64x(cur.nmod as i64), ex);
        let ge = _mm256_cmpgt_epi64(vnmod, vb2k_m1);
        vnmod = _mm256_sub_epi64(vnmod, _mm256_and_si256(ge, vb2k));
        let in_range = _mm256_cmpgt_epi64(vend, voff);
        let pass = _mm256_cmpeq_epi64(_mm256_and_si256(vlow, vhi), zero);
        let keep = _mm256_movemask_pd(_mm256_castsi256_pd(_mm256_and_si256(in_range, pass)));
        // SAFETY: the table has 16 rows and `keep` is a 4-bit mask; cnt <=
        // seen <= BLOCK + 3 and the buffers hold BLOCK + 16, so the four-lane
        // stores at `cnt` stay in bounds.
        unsafe {
            let perm =
                _mm256_loadu_si256(AVX2_COMPRESS.as_ptr().add(keep as usize).cast::<__m256i>());
            _mm256_storeu_si256(
                bufs.offs.as_mut_ptr().add(cnt).cast::<__m256i>(),
                _mm256_permutevar8x32_epi32(voff, perm),
            );
            _mm256_storeu_si256(
                bufs.nmods.as_mut_ptr().add(cnt).cast::<__m256i>(),
                _mm256_permutevar8x32_epi32(vnmod, perm),
            );
            _mm256_storeu_si256(
                bufs.lows.as_mut_ptr().add(cnt).cast::<__m256i>(),
                _mm256_permutevar8x32_epi32(vlow, perm),
            );
        }
        cnt += keep.count_ones() as usize;
        // Block total = last lane of the inclusive prefix.
        let total = _mm256_extract_epi64::<3>(g) as u64;
        cur.off += total;
        cur.nmod += total;
        if cur.nmod >= b2k {
            cur.nmod -= b2k;
        }
        cur.idx += 4;
        if cur.idx == glen {
            cur.idx = 0;
        }
        seen += 4;
    }
    (cnt, cur)
}

/// AVX-512 phase 1: eight candidates per step. The eight gaps are
/// prefix-summed in-register to give each lane its offset and its
/// `n mod b^{2k}`, the eight low masks are tested against the certificate, and the
/// surviving lanes are compressed (in register — the memory form of the
/// compress store is microcoded and slow) and stored unmasked at the
/// buffer cursor, which advances by the survivor count. The residue table's
/// last few entries before the wrap are walked by the scalar loop.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[allow(
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation,
    clippy::cast_ptr_alignment, // the loads are the unaligned forms
    clippy::too_many_lines
)]
unsafe fn phase1_avx512(
    gaps: &[u32],
    masks: &[u64],
    mut cur: Cursor,
    end_off: u64,
    b2k: u64,
    high_mask: u64,
    bufs: &mut SurvivorBufs,
) -> (usize, Cursor) {
    use std::arch::x86_64::{
        __m256i, __m512i, _mm256_extract_epi64, _mm256_loadu_si256, _mm512_add_epi64,
        _mm512_and_si512, _mm512_cmpeq_epi64_mask, _mm512_cmpge_epu64_mask,
        _mm512_cmplt_epu64_mask, _mm512_cvtepu32_epi64, _mm512_extracti64x4_epi64,
        _mm512_loadu_si512, _mm512_mask_sub_epi64, _mm512_maskz_compress_epi64,
        _mm512_maskz_permutexvar_epi64, _mm512_set_epi64, _mm512_set1_epi64, _mm512_setzero_si512,
        _mm512_storeu_si512,
    };
    let glen = gaps.len();
    let mut cnt = 0usize;
    let mut seen = 0usize;
    let vhi = _mm512_set1_epi64(high_mask as i64);
    let vb2k = _mm512_set1_epi64(b2k as i64);
    let vend = _mm512_set1_epi64(end_off as i64);
    let zero = _mm512_setzero_si512();
    // Lane shifts for the in-register inclusive prefix sum.
    let sh1 = _mm512_set_epi64(6, 5, 4, 3, 2, 1, 0, 0);
    let sh2 = _mm512_set_epi64(5, 4, 3, 2, 1, 0, 0, 0);
    let sh4 = _mm512_set_epi64(3, 2, 1, 0, 0, 0, 0, 0);
    while cur.off < end_off && seen < BLOCK {
        if cur.idx + 8 > glen {
            // Tail of the residue table before the wrap: scalar.
            let n = glen - cur.idx;
            let (c, next) = phase1_scalar(gaps, masks, cur, end_off, b2k, high_mask, bufs, cnt, n);
            cnt = c;
            cur = next;
            seen += n;
            continue;
        }
        // SAFETY: cur.idx + 8 <= glen == masks.len(); loads stay in bounds.
        let (g32, vlow) = unsafe {
            (
                _mm256_loadu_si256(gaps.as_ptr().add(cur.idx).cast::<__m256i>()),
                _mm512_loadu_si512(masks.as_ptr().add(cur.idx).cast::<__m512i>()),
            )
        };
        let mut g = _mm512_cvtepu32_epi64(g32);
        g = _mm512_add_epi64(g, _mm512_maskz_permutexvar_epi64(0b1111_1110, sh1, g));
        g = _mm512_add_epi64(g, _mm512_maskz_permutexvar_epi64(0b1111_1100, sh2, g));
        g = _mm512_add_epi64(g, _mm512_maskz_permutexvar_epi64(0b1111_0000, sh4, g));
        // Exclusive prefix: lane i holds the sum of gaps before candidate i.
        let ex = _mm512_maskz_permutexvar_epi64(0b1111_1110, sh1, g);
        let voff = _mm512_add_epi64(_mm512_set1_epi64(cur.off as i64), ex);
        let mut vnmod = _mm512_add_epi64(_mm512_set1_epi64(cur.nmod as i64), ex);
        let ge = _mm512_cmpge_epu64_mask(vnmod, vb2k);
        vnmod = _mm512_mask_sub_epi64(vnmod, ge, vnmod, vb2k);
        let in_range = _mm512_cmplt_epu64_mask(voff, vend);
        let pass = _mm512_cmpeq_epi64_mask(_mm512_and_si512(vlow, vhi), zero);
        let keep = in_range & pass;
        // SAFETY: cnt <= seen <= BLOCK + 7 and the buffers hold BLOCK + 16,
        // so the eight-lane stores at `cnt` stay in bounds.
        unsafe {
            _mm512_storeu_si512(
                bufs.offs.as_mut_ptr().add(cnt).cast::<__m512i>(),
                _mm512_maskz_compress_epi64(keep, voff),
            );
            _mm512_storeu_si512(
                bufs.nmods.as_mut_ptr().add(cnt).cast::<__m512i>(),
                _mm512_maskz_compress_epi64(keep, vnmod),
            );
            _mm512_storeu_si512(
                bufs.lows.as_mut_ptr().add(cnt).cast::<__m512i>(),
                _mm512_maskz_compress_epi64(keep, vlow),
            );
        }
        cnt += keep.count_ones() as usize;
        // Block total = last lane of the inclusive prefix.
        let total = _mm256_extract_epi64::<3>(_mm512_extracti64x4_epi64::<1>(g)) as u64;
        cur.off += total;
        cur.nmod += total;
        if cur.nmod >= b2k {
            cur.nmod -= b2k;
        }
        cur.idx += 8;
        if cur.idx == glen {
            cur.idx = 0;
        }
        seen += 8;
    }
    (cnt, cur)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base_range::get_base_range_u128;
    use crate::client_process::get_is_nice;
    use crate::msd_prefix_filter::get_valid_ranges_masked;

    /// Base-`base` digits of `n²` and `n³`, least significant first.
    fn power_digits(n: u128, base: u32) -> (Vec<u32>, Vec<u32>) {
        let mut sq = crate::fixed_width::U256::mul_u128_u128(n, n);
        let mut cu = sq.mul_u128_truncating(n);
        let (mut sqd, mut cud) = (Vec::new(), Vec::new());
        while !sq.is_zero() {
            sqd.push(sq.div_assign_rem_u32(base));
        }
        while !cu.is_zero() {
            cud.push(cu.div_assign_rem_u32(base));
        }
        (sqd, cud)
    }

    /// The masked walk with the affine filter lets through exactly the
    /// cross-end survivors whose *true* middle digits (positions 3..5 of
    /// `n²` and `n³`, from wide arithmetic) are pairwise distinct and miss
    /// the residue's low digits and the range certificate. Checked on every
    /// candidate of real MSD leaves, over windows long enough to wrap the
    /// gap table, so a drift in the walk's incremental `n mod b^6` shows up
    /// as a disagreement rather than as an empty nice set on both sides.
    #[test_log::test]
    fn affine_walk_decisions_match_true_digits() {
        // The benchmark's MSD-weak windows (`bench_defs`): production
        // regions with plenty of cross-end survivors.
        for (base, start) in [
            (40u32, 5_007_828_088_304u128),
            (50, 73_940_161_512_353_211),
            (52, 407_887_399_136_188_818),
        ] {
            assert!(affine_filter::supports(base, 3));
            let table = StrideTable::new(base, 3);
            // Longer than the gap table's period M = (b-1)·b³, so every leaf
            // sequence crosses the table's wrap at least once.
            let len = table.modulus + 1_000_000;
            let range = FieldSize::new(start, start + len);
            let (mut crossed, mut passed) = (0usize, 0usize);
            for (leaf, high) in crate::msd_prefix_filter::get_valid_ranges_masked(range, base, 3) {
                let mut cross = Vec::new();
                let _ = table.walk_masked(&leaf, base, high, false, |n, low| {
                    cross.push((n, low));
                    false
                });
                let mut through = Vec::new();
                let _ = table.walk_masked(&leaf, base, high, true, |n, _| {
                    through.push(n);
                    false
                });
                let mut it = through.iter().peekable();
                for (n, low) in cross {
                    let (sq, cu) = power_digits(n, base);
                    let mut seen = low | high;
                    let mut fresh = true;
                    for &d in sq[3..6].iter().chain(&cu[3..6]) {
                        fresh &= seen & (1u64 << d) == 0;
                        seen |= 1u64 << d;
                    }
                    let got = it.next_if_eq(&&n).is_some();
                    assert_eq!(got, fresh, "b{base} n={n}: walk and true digits disagree");
                    crossed += 1;
                    passed += usize::from(got);
                }
                assert!(
                    it.next().is_none(),
                    "b{base}: filtered walk visited extra candidates"
                );
            }
            // Not vacuous: plenty of cross-end survivors, most of them killed.
            assert!(
                crossed > 5_000,
                "b{base}: only {crossed} cross-end survivors"
            );
            assert!(
                passed * 10 < crossed,
                "b{base}: {passed} of {crossed} passed the affine filter"
            );
        }
    }

    #[test_log::test]
    fn test_stride_table_base10_k1() {
        let table = StrideTable::new(10, 1);

        // Base 10: (b-1) = 9, b^1 = 10, M = 90
        assert_eq!(table.modulus, 90);

        // Should have valid residues combining both filters
        assert!(!table.valid_residues.is_empty(), "no valid residues");
        assert_eq!(table.valid_residues.len(), table.gap_table.len());

        // Verify gap table covers full cycle
        let total_gap: u128 = table.gap_table.iter().map(|&g| u128::from(g)).sum();
        assert_eq!(total_gap, table.modulus);
    }

    #[test_log::test]
    fn test_stride_table_base40_k2() {
        let table = StrideTable::new(40, 2);

        // Base 40: (b-1) = 39, b^2 = 1600, M = 62400
        assert_eq!(table.modulus, 62_400);

        // Should filter significantly
        assert!(table.valid_residues.len() < (table.modulus as usize));

        // Verify properties
        assert_eq!(table.valid_residues.len(), table.gap_table.len());
        let total_gap: u128 = table.gap_table.iter().map(|&g| u128::from(g)).sum();
        assert_eq!(total_gap, table.modulus);
    }

    #[test_log::test]
    fn test_first_valid_at_or_after() {
        let table = StrideTable::new(10, 1);

        // Start at 0 should find first valid
        let (n, idx) = table.first_valid_at_or_after(0);
        assert_eq!(n, u128::from(table.valid_residues[idx]));

        // Start at a valid residue should return it
        let first_valid = u128::from(table.valid_residues[0]);
        let (n, idx) = table.first_valid_at_or_after(first_valid);
        assert_eq!(n, first_valid);
        assert_eq!(idx, 0);

        // Start beyond modulus should wrap correctly
        let (n, idx) = table.first_valid_at_or_after(table.modulus + 5);
        assert!(n >= table.modulus + 5);
        assert_eq!(n % table.modulus, u128::from(table.valid_residues[idx]));
    }

    #[test_log::test]
    fn test_stride_iteration_finds_known_nice() {
        // Base 10: 69 is a known nice number
        let table = StrideTable::new(10, 1);

        let range = FieldSize::new(60, 80);
        let results = table.iterate_range(&range, 10);

        // Should find 69
        assert!(results.iter().any(|r| r.number == 69));
    }

    #[test_log::test]
    fn test_k3_candidates_subset_of_k2() {
        // Deeper LSD filtering must only remove candidates, never add them:
        // all 3+3 fixed low digits distinct implies the low 2+2 subset is
        // distinct, so every k=3 candidate must also be a k=2 candidate.
        for base in [10u32, 40] {
            let t2 = StrideTable::new(base, 2);
            let t3 = StrideTable::new(base, 3);
            let start = 1_000_000u128;
            let range = FieldSize::new(start, start + 200_000);

            let collect = |t: &StrideTable| {
                let mut out = Vec::new();
                let (mut n, mut idx) = t.first_valid_at_or_after(range.start());
                while n < range.end() {
                    out.push(n);
                    n += u128::from(t.gap_table[idx]);
                    idx = (idx + 1) % t.gap_table.len();
                }
                out
            };
            let c2 = collect(&t2);
            let c3 = collect(&t3);
            assert!(c3.len() < c2.len(), "base {base}: k=3 should filter more");
            let c2set: std::collections::HashSet<u128> = c2.into_iter().collect();
            for n in c3 {
                assert!(c2set.contains(&n), "base {base}: {n} in k=3 but not k=2");
            }
        }
    }

    #[test_log::test]
    fn test_k3_finds_known_nice() {
        // 69 must survive the k=3 table in base 10
        let table = StrideTable::new(10, 3);
        let range = FieldSize::new(60, 80);
        let results = table.iterate_range(&range, 10);
        assert!(results.iter().any(|r| r.number == 69));
    }

    type Phase1Fn = fn(&[u32], &[u64], Cursor, u64, u64, u64, &mut SurvivorBufs) -> (usize, Cursor);

    /// Every phase-1 implementation this CPU can run: the dispatched one,
    /// scalar with full blocks, and AVX2 / AVX-512 where detected.
    fn phase1_impls() -> Vec<(&'static str, Phase1Fn)> {
        #[allow(unused_mut)]
        let mut impls: Vec<(&'static str, Phase1Fn)> = vec![
            ("dispatch", phase1),
            ("scalar", |g, m, c, e, b, h, bufs| {
                phase1_scalar(g, m, c, e, b, h, bufs, 0, BLOCK)
            }),
        ];
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx2") {
                impls.push(("avx2", |g, m, c, e, b, h, bufs| unsafe {
                    phase1_avx2(g, m, c, e, b, h, bufs)
                }));
            }
            if std::arch::is_x86_feature_detected!("avx512f") {
                impls.push(("avx512", |g, m, c, e, b, h, bufs| unsafe {
                    phase1_avx512(g, m, c, e, b, h, bufs)
                }));
            }
        }
        impls
    }

    /// Phase 1 must hand phase 2 exactly the cross-end survivors a plain
    /// walk finds, in count and content, through every implementation the
    /// CPU can run, on production windows of every specialized base,
    /// including the residue table's wrap.
    #[test_log::test]
    #[allow(clippy::cast_possible_truncation)]
    fn two_phase_walk_matches_plain_walk() {
        for base in [40u32, 42, 45, 50, 52, 53, 57, 60, 62, 64] {
            let table = StrideTable::new(base, 3);
            let range = get_base_range_u128(base).unwrap().unwrap();
            // The MSD filter rejects some windows outright; take the first
            // of a few evenly spaced ones that leaves work for the walk.
            let leaves = (1..200)
                .map(|step| {
                    let start = range.start() + range.size() / 200 * step;
                    get_valid_ranges_masked(FieldSize::new(start, start + 3_000_000), base, 3)
                })
                .find(|leaves| !leaves.is_empty())
                .unwrap_or_else(|| panic!("b{base}: every probed window fully rejected"));
            let bk = u64::from(base).pow(3);
            let b2k = bk * bk;
            let mut checked = 0usize;
            for (leaf, hi) in leaves {
                // Plain survivor list.
                let mut want: Vec<(u64, u64, u64)> = Vec::new();
                let (mut n, mut idx) = table.first_valid_at_or_after(leaf.start());
                while n < leaf.end() {
                    let low = table.low_digit_masks[idx];
                    if low & hi == 0 {
                        want.push(((n - leaf.start()) as u64, (n % u128::from(b2k)) as u64, low));
                    }
                    n += u128::from(table.gap_table[idx]);
                    idx = (idx + 1) % table.gap_table.len();
                }
                // Block-wise phase 1 through every implementation this CPU
                // can run, the dispatched one included.
                let (n0, idx0) = table.first_valid_at_or_after(leaf.start());
                let end_off = (leaf.end() - leaf.start()) as u64;
                let start_cursor = Cursor {
                    idx: idx0,
                    off: (n0 - leaf.start()) as u64,
                    nmod: (n0 % u128::from(b2k)) as u64,
                };
                let mut bufs = SurvivorBufs::new();
                for (name, imp) in &phase1_impls() {
                    let mut cur = start_cursor;
                    let mut got: Vec<(u64, u64, u64)> = Vec::new();
                    while cur.off < end_off {
                        let (cnt, next) = imp(
                            &table.gap_table,
                            &table.low_digit_masks,
                            cur,
                            end_off,
                            b2k,
                            hi,
                            &mut bufs,
                        );
                        for i in 0..cnt {
                            got.push((bufs.offs[i], bufs.nmods[i], bufs.lows[i]));
                        }
                        cur = next;
                    }
                    assert_eq!(got, want, "b{base} leaf {leaf:?} ({name})");
                }
                let mut cur = start_cursor;
                // And the scalar implementation on its own, in small blocks so
                // block boundaries and the wrap are both exercised.
                let mut got_scalar: Vec<(u64, u64, u64)> = Vec::new();
                while cur.off < end_off {
                    let (cnt, next) = phase1_scalar(
                        &table.gap_table,
                        &table.low_digit_masks,
                        cur,
                        end_off,
                        b2k,
                        hi,
                        &mut bufs,
                        0,
                        13,
                    );
                    for i in 0..cnt {
                        got_scalar.push((bufs.offs[i], bufs.nmods[i], bufs.lows[i]));
                    }
                    cur = next;
                }
                assert_eq!(got_scalar, want, "b{base} leaf {leaf:?} (scalar)");
                assert_eq!(table.two_phase_survivor_count(&leaf, base, hi), want.len());
                checked += want.len();
            }
            assert!(checked > 100, "b{base}: too few survivors exercised");
        }
    }

    /// Survivor buffers between canaries, so a store past either end shows.
    #[repr(C)]
    struct Guarded {
        pre: [u64; 16],
        bufs: SurvivorBufs,
        post: [u64; 64],
    }

    const CANARY: u64 = 0xA5A5_5A5A_DEAD_BEEF;

    impl Guarded {
        fn new() -> Box<Self> {
            Box::new(Self {
                pre: [CANARY; 16],
                bufs: SurvivorBufs::new(),
                post: [CANARY; 64],
            })
        }
    }

    /// Drive `imp` over `[cur, end_off)`, poisoning the buffers before every
    /// block (so a stale read shows as a sentinel) and checking the canaries
    /// and the cursor after it.
    #[allow(clippy::too_many_arguments)]
    fn drive(
        imp: Phase1Fn,
        table: &StrideTable,
        mut cur: Cursor,
        end_off: u64,
        b2k: u64,
        hi: u64,
        g: &mut Guarded,
    ) -> Vec<(u64, u64, u64)> {
        let mut got = Vec::new();
        while cur.off < end_off {
            g.bufs.offs.fill(u64::MAX);
            g.bufs.nmods.fill(u64::MAX);
            g.bufs.lows.fill(u64::MAX);
            let (cnt, next) = imp(
                &table.gap_table,
                &table.low_digit_masks,
                cur,
                end_off,
                b2k,
                hi,
                &mut g.bufs,
            );
            assert!(
                g.pre.iter().chain(&g.post).all(|&c| c == CANARY),
                "canary clobbered"
            );
            assert!(next.off > cur.off, "no progress");
            assert!(next.idx < table.gap_table.len());
            assert!(next.nmod < b2k, "n mod b^6 not reduced");
            // A SIMD step may finish up to 7 candidates past the block.
            assert!(cnt <= BLOCK + 7);
            got.extend((0..cnt).map(|i| (g.bufs.offs[i], g.bufs.nmods[i], g.bufs.lows[i])));
            cur = next;
        }
        got
    }

    /// Cross-end survivors of `leaf` by plain stepping and `%`.
    #[allow(clippy::cast_possible_truncation)]
    fn plain_survivors(
        table: &StrideTable,
        leaf: &FieldSize,
        b2k: u64,
        hi: u64,
    ) -> Vec<(u64, u64, u64)> {
        let mut out = Vec::new();
        let (mut n, mut idx) = table.first_valid_at_or_after(leaf.start());
        while n < leaf.end() {
            let low = table.low_digit_masks[idx];
            if low & hi == 0 {
                out.push(((n - leaf.start()) as u64, (n % u128::from(b2k)) as u64, low));
            }
            n += u128::from(table.gap_table[idx]);
            idx = (idx + 1) % table.gap_table.len();
        }
        out
    }

    /// Every implementation, scalar in `small_block`-candidate blocks, and
    /// phase 2 (through `walk_masked` with a recorder in place of the nice
    /// check) against the plain walk, on one (leaf, hi).
    #[allow(clippy::cast_possible_truncation)]
    fn check_leaf(
        table: &StrideTable,
        leaf: &FieldSize,
        base: u32,
        hi: u64,
        small_block: usize,
        g: &mut Guarded,
    ) -> usize {
        let b2k = u64::from(base).pow(6);
        let want = plain_survivors(table, leaf, b2k, hi);
        let end_off = (leaf.end() - leaf.start()) as u64;
        let (n0, idx0) = table.first_valid_at_or_after(leaf.start());
        let c0 = Cursor {
            idx: idx0,
            off: (n0 - leaf.start()) as u64,
            nmod: (n0 % u128::from(b2k)) as u64,
        };
        for (name, imp) in phase1_impls() {
            assert_eq!(
                drive(imp, table, c0, end_off, b2k, hi, g),
                want,
                "b{base} {leaf:?} hi={hi:#x} ({name})"
            );
        }
        let mut cur = c0;
        let mut got = Vec::new();
        while cur.off < end_off {
            let (cnt, next) = phase1_scalar(
                &table.gap_table,
                &table.low_digit_masks,
                cur,
                end_off,
                b2k,
                hi,
                &mut g.bufs,
                0,
                small_block,
            );
            got.extend((0..cnt).map(|i| (g.bufs.offs[i], g.bufs.nmods[i], g.bufs.lows[i])));
            cur = next;
        }
        assert_eq!(
            got, want,
            "b{base} {leaf:?} hi={hi:#x} (scalar, blocks of {small_block})"
        );
        let mut checked = Vec::new();
        let _ = table.walk_masked(leaf, base, hi, true, |n, _| {
            checked.push(n);
            false
        });
        let want2: Vec<u128> = want
            .iter()
            .filter(|&&(_, nmod, low)| affine_filter::survives(base, nmod, low | hi))
            .map(|&(off, _, _)| leaf.start() + u128::from(off))
            .collect();
        assert_eq!(checked, want2, "b{base} {leaf:?} hi={hi:#x}: phase 2");
        want.len()
    }

    /// A small xorshift, so the tests are reproducible.
    struct Rng(u64);

    impl Rng {
        fn below(&mut self, n: u64) -> u64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) % n
        }
    }

    /// Start `j` residues before the end of the residue table, for every `j`
    /// up to a block and a bit, so the table's wrap lands on every SIMD lane
    /// and on the block boundary; end exactly on a candidate (excluded) and
    /// one past it; with no, one and several certificate digits.
    #[test_log::test]
    fn phase1_matches_plain_at_every_wrap_and_block_alignment() {
        let mut g = Guarded::new();
        for base in [40u32, 50, 64] {
            let table = StrideTable::new(base, 3);
            let glen = table.gap_table.len();
            let m = table.modulus;
            let range = get_base_range_u128(base).unwrap().unwrap();
            let period = (range.start() / m + 7) * m - m;
            for j in 1..=(BLOCK + 20) {
                let start = period + u128::from(table.valid_residues[glen - j]);
                let mut idx = glen - j;
                let mut end = start;
                for _ in 0..(j + 40) {
                    end += u128::from(table.gap_table[idx]);
                    idx = (idx + 1) % glen;
                }
                let several = 0x0F0F_0F0F & (u64::MAX >> (64 - base));
                for hi in [0u64, 1u64 << (j as u64 % u64::from(base)), several] {
                    for extra in [0u128, 1] {
                        let leaf = FieldSize::new(start, end + extra);
                        check_leaf(&table, &leaf, base, hi, 1 + j % 40, &mut g);
                    }
                }
            }
        }
    }

    /// Random ranges on every base that takes the two-phase walk: starts at,
    /// just before and well before a multiple of the stride modulus, lengths
    /// from one number to one and a half periods, any certificate.
    #[test_log::test]
    fn phase1_matches_plain_on_random_ranges_of_every_base() {
        let mut g = Guarded::new();
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        let bases: Vec<u32> = (10..=64u32)
            .filter(|&b| affine_filter::supports(b, 3))
            .filter(|&b| !StrideTable::new(b, 3).gap_table.is_empty())
            .collect();
        assert_eq!(bases.len(), 15);
        let mut total = 0;
        for &base in &bases {
            let table = StrideTable::new(base, 3);
            let m = table.modulus;
            let m64 = u64::try_from(m).unwrap();
            let range = get_base_range_u128(base).unwrap().unwrap();
            for _ in 0..12 {
                let anywhere =
                    range.start() + u128::from(rng.below(u64::MAX)) % (range.size() - 100 * m);
                let next_period = (anywhere / m + 1) * m;
                let start = match rng.below(4) {
                    0 => anywhere,
                    1 => next_period,
                    2 => next_period - u128::from(rng.below(m64 / 50 + 1)),
                    _ => next_period - u128::from(rng.below(m64 / 5 + 1)),
                };
                let len = match rng.below(4) {
                    0 => 1 + u128::from(rng.below(3)),
                    1 => 1 + u128::from(rng.below(40_000)),
                    2 => 1 + u128::from(rng.below(m64 + m64 / 2)),
                    _ => 1 + u128::from(rng.below(300_000)),
                };
                let hi = match rng.below(4) {
                    0 => 0,
                    1 => u64::MAX >> (64 - base),
                    2 => 1u64 << rng.below(u64::from(base)),
                    _ => (0..=rng.below(10))
                        .fold(0u64, |h, _| h | 1u64 << rng.below(u64::from(base))),
                };
                let small_block = 1 + usize::try_from(rng.below(40)).unwrap();
                total += check_leaf(
                    &table,
                    &FieldSize::new(start, start + len),
                    base,
                    hi,
                    small_block,
                    &mut g,
                );
            }
        }
        assert!(total > 10_000, "only {total} survivors compared");
    }

    /// Offsets at the top of the permitted range (a range under `2^62`).
    #[test_log::test]
    fn phase1_matches_plain_at_large_offsets() {
        let mut g = Guarded::new();
        let mut rng = Rng(777);
        for base in [40u32, 57, 64] {
            let table = StrideTable::new(base, 3);
            let glen = table.gap_table.len();
            let b2k = u64::from(base).pow(6);
            for _ in 0..100 {
                let end_off = (1u64 << 62) - 1 - rng.below(3);
                let cur = Cursor {
                    idx: usize::try_from(rng.below(glen as u64)).unwrap(),
                    off: end_off - 1 - rng.below(100_000),
                    nmod: rng.below(b2k),
                };
                let hi = if rng.below(2) == 0 {
                    0
                } else {
                    1u64 << rng.below(u64::from(base))
                };
                let mut want = Vec::new();
                let mut c = cur;
                while c.off < end_off {
                    let low = table.low_digit_masks[c.idx];
                    if low & hi == 0 {
                        want.push((c.off, c.nmod, low));
                    }
                    let gap = u64::from(table.gap_table[c.idx]);
                    c.off += gap;
                    c.nmod = (c.nmod + gap) % b2k;
                    c.idx = (c.idx + 1) % glen;
                }
                for (name, imp) in phase1_impls() {
                    assert_eq!(
                        drive(imp, &table, cur, end_off, b2k, hi, &mut g),
                        want,
                        "b{base} ({name})"
                    );
                }
            }
        }
    }

    #[test_log::test]
    fn test_seeded_nice_check_matches_plain() {
        // The seeded fast path must agree with the plain check for every
        // stride candidate. Walk real candidates inside each base's search
        // range and compare both paths.
        for base in [40u32, 50, 52, 60, 64] {
            let table = StrideTable::new(base, 3);
            assert!(!table.low_digit_masks.is_empty(), "b{base}: no masks");
            let range = get_base_range_u128(base).unwrap().unwrap();
            let start = range.start() + (range.end() - range.start()) / 3;
            let (mut n, mut idx) = table.first_valid_at_or_after(start);
            for _ in 0..5_000 {
                let plain = get_is_nice(n, base);
                let seeded =
                    get_is_nice_with_known_lsd(n, base, table.k, table.low_digit_masks[idx]);
                assert_eq!(plain, seeded, "base {base}: mismatch at n={n}");
                n += u128::from(table.gap_table[idx]);
                idx = (idx + 1) % table.gap_table.len();
            }
        }
    }

    #[test_log::test]
    fn test_low_digit_masks_have_2k_bits() {
        // Every mask covers exactly 2k distinct digits (k from the square
        // suffix, k from the cube suffix, all pairwise distinct).
        for (base, k) in [(10u32, 2u32), (40, 3), (50, 3)] {
            let table = StrideTable::new(base, k);
            for (&r, &mask) in table.valid_residues.iter().zip(&table.low_digit_masks) {
                assert_eq!(
                    mask.count_ones(),
                    2 * k,
                    "base {base} k={k} residue {r}: expected {} distinct digits",
                    2 * k
                );
            }
        }
    }

    #[test_log::test]
    fn test_k3_large_base_construction() {
        // Largest production-adjacent table: base 60, k=3.
        // M = 59 * 60^3 = 12,744,000 and all residues/gaps must fit u32.
        let table = StrideTable::new(60, 3);
        assert_eq!(table.modulus, 12_744_000);
        assert!(!table.valid_residues.is_empty(), "no valid residues");
        let total_gap: u128 = table.gap_table.iter().map(|&g| u128::from(g)).sum();
        assert_eq!(total_gap, table.modulus);
    }

    #[test_log::test]
    fn test_gap_table_properties() {
        let table = StrideTable::new(10, 2);

        // All gaps should be positive
        for gap in &table.gap_table {
            assert!(*gap > 0, "Gap should be positive");
        }

        // Sum of gaps should equal modulus (complete cycle)
        let total: u128 = table.gap_table.iter().map(|&g| u128::from(g)).sum();
        assert_eq!(total, table.modulus);

        // Valid residues should be sorted
        for i in 1..table.valid_residues.len() {
            assert!(
                table.valid_residues[i] > table.valid_residues[i - 1],
                "Valid residues should be sorted"
            );
        }
    }
}
