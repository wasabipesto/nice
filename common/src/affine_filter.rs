//! Affine middle-digit filter for cross-end survivors.
//!
//! Write a candidate as `n = s + B·t` with `B = b^k` (the stride table's LSD
//! depth, k = 3 in production), so `s = n mod B` is the residue's suffix and
//! `t = n div B`. Then
//!
//! ```text
//! n² = s² + 2sB·t + B²t²          n³ = s³ + 3s²B·t + 3sB²t² + B³t³
//! ```
//!
//! Modulo `B² = b^{2k}` every term carrying `B²` vanishes, so the low `2k`
//! digits of both powers depend only on `s` and `t mod B`:
//!
//! ```text
//! n² mod B² = s² + B·(2s·t mod B)          (mod B²)
//! n³ mod B² = s³ + B·(3s²·t mod B)         (mod B²)
//! ```
//!
//! The lowest `k` of those digits are the stride table's exact low digits,
//! already tested. The next `k` — output positions `k..2k-1` of each power,
//! the first positions the seeded nice check would otherwise have to
//! discover with wide arithmetic — are the base-`b` digits of
//!
//! ```text
//! q = (⌊s²/B⌋ + 2s·t) mod B        c = (⌊s³/B⌋ + 3s²·t) mod B
//! ```
//!
//! For `b ≤ 64` and `k = 3` everything fits in `u64` (`s³ < 2^54`,
//! `3s²t < 2^56`) and every divisor is a compile-time constant, so the six
//! digits cost a few dozen instructions with no multi-limb work.
//!
//! A candidate that has passed the cross-end test carries a `known` mask of
//! digits certainly present elsewhere in the output: its residue's `2k` low
//! digits plus the range certificate's high digits. Each fresh digit collides
//! with it with probability about `|known|/b ≈ 0.4`, so the six digits here
//! reject roughly 97% of cross-end survivors before the full check runs
//! (measured 96.6-97.3% on bases 40-60).
//!
//! Soundness: the filter rejects only when two of the six digits coincide
//! or one of them is in `known`. Both are genuine repeats, which no nice
//! number has, provided that
//!
//! 1. the six positions are real digits of `n²` and `n³`, and
//! 2. every digit of `known` occurs in `n²` or `n³` at some position other
//!    than the six tested here.
//!
//! (1) fails only when `n²` has fewer than six digits (`n² < b^5`); then
//! `n³ < b^8`, so the two powers have at most 13 digits between them and
//! `n` cannot be nice in any base the filter supports (`b ≥ 40` digits are
//! needed). A zero-padded position can only ever reject a number that was
//! not nice anyway.
//!
//! (2) holds for the residue's low digits by construction (positions
//! `0..k`). The range certificate holds the digits at positions `j ≥ k`
//! that are the same for every `n` in an analysed range; the MSD recursion
//! only analyses ranges larger than its floor (at least 1), so such a range
//! holds some `x` and `x ± 1`, and a position `j` can only be the same for
//! both if `b^j > 2x - 1`. The stride walk enables the filter only for
//! ranges starting at or above `b^5`, where that rules out positions 3..5
//! (see `StrideTable::iterate_range_masked`); every legal range of every
//! supported base starts far above it (pinned by the test
//! `legal_ranges_start_above_the_gate`). The gate also puts `n²` at eleven
//! digits or more, so (1) never even arises. A caller that builds `known`
//! any other way must keep positions 3..5 out of it.

/// Whether the affine filter has a sound, specialized path for this base
/// and stride depth. When this is false, callers must skip the filter.
#[must_use]
pub const fn supports(base: u32, k: u32) -> bool {
    k == 3 && matches!(base, 40 | 42..=45 | 47..=50 | 52..=55 | 57..=60 | 62 | 64)
}

/// Test the six middle digits of `n²` and `n³` against `known`.
///
/// `nmod` is `n mod b^{2k}` and `known` the union of the residue's low-digit
/// mask and the range's high certificate, whose digits must all sit at
/// positions other than the six middle ones (see the module docs for why
/// the MSD certificate satisfies this). Returns `true` when the candidate
/// survives, i.e. the six digits are pairwise distinct and avoid `known`.
///
/// Callers must check [`supports`] first; other bases return `true`
/// (no filtering).
#[must_use]
#[inline]
pub fn survives(base: u32, nmod: u64, known: u64) -> bool {
    match base {
        40 => survives_const::<40>(nmod, known),
        42 => survives_const::<42>(nmod, known),
        43 => survives_const::<43>(nmod, known),
        44 => survives_const::<44>(nmod, known),
        45 => survives_const::<45>(nmod, known),
        47 => survives_const::<47>(nmod, known),
        48 => survives_const::<48>(nmod, known),
        49 => survives_const::<49>(nmod, known),
        50 => survives_const::<50>(nmod, known),
        52 => survives_const::<52>(nmod, known),
        53 => survives_const::<53>(nmod, known),
        54 => survives_const::<54>(nmod, known),
        55 => survives_const::<55>(nmod, known),
        57 => survives_const::<57>(nmod, known),
        58 => survives_const::<58>(nmod, known),
        59 => survives_const::<59>(nmod, known),
        60 => survives_const::<60>(nmod, known),
        62 => survives_const::<62>(nmod, known),
        64 => survives_const::<64>(nmod, known),
        _ => true,
    }
}

/// Compile-time-base body of [`survives`]: every `/` and `%` below is by a
/// constant and lowers to multiply-high sequences. Branchless.
#[inline]
fn survives_const<const BASE: u32>(nmod: u64, known: u64) -> bool {
    const { assert!(BASE <= 64, "digit masks are u64") };
    const { assert!(BASE >= 15, "n² must have at least 6 digits in range") };
    let (mq, mc) = middle_masks_const::<BASE>(nmod);
    let dup = u64::from(mq.count_ones() != 3) | u64::from(mc.count_ones() != 3);
    (((mq | mc) & known) | (mq & mc) | dup) == 0
}

/// Digit masks of output positions `3..6` of `n²` (first) and `n³` (second),
/// from `nmod = n mod BASE^6`. A repeated digit within one power shows up as
/// a mask with fewer than three bits.
#[inline]
#[allow(clippy::similar_names)]
fn middle_masks_const<const BASE: u32>(nmod: u64) -> (u64, u64) {
    let base = u64::from(BASE);
    let bk = const { (BASE as u64) * (BASE as u64) * (BASE as u64) };
    let suffix = nmod % bk;
    let quot = nmod / bk;
    let suffix2 = suffix * suffix; // < b^6 < 2^36
    let suffix3 = suffix2 * suffix; // < b^9 < 2^54
    let sq_mid = (suffix2 / bk + 2 * suffix * quot) % bk; // 2st < 2·b^6
    let cu_mid = (suffix3 / bk + 3 * suffix2 * quot) % bk; // 3s²t < 3·b^9 < 2^56
    let sq0 = sq_mid % base;
    let sq12 = sq_mid / base;
    let sq1 = sq12 % base;
    let sq2 = sq12 / base;
    let cu0 = cu_mid % base;
    let cu12 = cu_mid / base;
    let cu1 = cu12 % base;
    let cu2 = cu12 / base;
    let mask_sq = (1u64 << sq0) | (1u64 << sq1) | (1u64 << sq2);
    let mask_cu = (1u64 << cu0) | (1u64 << cu1) | (1u64 << cu2);
    (mask_sq, mask_cu)
}

/// The six middle digits themselves, for tests and the GPU kernels' host
/// mirrors: `(n² digits at positions 3,4,5, n³ digits at positions 3,4,5)`.
/// Requires `2 ≤ base ≤ 64` and `nmod < base^6`, so that `nmod³` fits
/// `u64`.
#[cfg(test)]
#[must_use]
pub(crate) fn middle_digits(base: u32, nmod: u64) -> ([u32; 3], [u32; 3]) {
    debug_assert!((2..=64).contains(&base));
    debug_assert!(u128::from(nmod) < u128::from(base).pow(6));
    let base = u64::from(base);
    let bk = base * base * base;
    let suffix = nmod % bk;
    let quot = nmod / bk;
    let suffix2 = suffix * suffix;
    let suffix3 = suffix2 * suffix;
    let mut sq_mid = (suffix2 / bk + 2 * suffix * quot) % bk;
    let mut cu_mid = (suffix3 / bk + 3 * suffix2 * quot) % bk;
    let mut sq = [0u32; 3];
    let mut cu = [0u32; 3];
    for i in 0..3 {
        #[allow(clippy::cast_possible_truncation)]
        {
            sq[i] = (sq_mid % base) as u32;
            cu[i] = (cu_mid % base) as u32;
        }
        sq_mid /= base;
        cu_mid /= base;
    }
    (sq, cu)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FieldSize;
    use crate::base_range::get_base_range_u128;
    use crate::fixed_width::U256;
    use crate::stride_filter::StrideTable;

    fn true_digits(n: u128, base: u32) -> (Vec<u32>, Vec<u32>) {
        let mut sq = U256::mul_u128_u128(n, n);
        let mut cu = sq.mul_u128_truncating(n);
        let mut sqd = Vec::new();
        let mut cud = Vec::new();
        while !sq.is_zero() {
            sqd.push(sq.div_assign_rem_u32(base));
        }
        while !cu.is_zero() {
            cud.push(cu.div_assign_rem_u32(base));
        }
        (sqd, cud)
    }

    fn lcg(state: &mut u128) -> u128 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        *state
    }

    #[test_log::test]
    fn middle_digits_match_true_digits_all_specialized_bases() {
        // The filter only ever rejects on a genuine duplicate among real
        // output digits, so digit equality with wide arithmetic is the
        // soundness proof.
        let bases: &[u32] = &[
            40, 42, 43, 44, 45, 47, 48, 49, 50, 52, 53, 54, 55, 57, 58, 59, 60, 62, 64,
        ];
        let mut state: u128 = 0x9e37_79b9_7f4a_7c15_f39c_c060_5ced_c834;
        for &base in bases {
            assert!(supports(base, 3));
            let r = get_base_range_u128(base).unwrap().unwrap();
            let b6 = u128::from(base).pow(6);
            for _ in 0..3000 {
                let n = r.start() + lcg(&mut state) % r.size();
                #[allow(clippy::cast_possible_truncation)]
                let nmod = (n % b6) as u64;
                let (sq, cu) = middle_digits(base, nmod);
                let (tsq, tcu) = true_digits(n, base);
                assert!(
                    tsq.len() >= 6 && tcu.len() >= 6,
                    "b{base}: powers too short"
                );
                assert_eq!(sq, tsq[3..6], "b{base} n={n}: square middle digits");
                assert_eq!(cu, tcu[3..6], "b{base} n={n}: cube middle digits");
                // And the verdict is exactly "six real digits, distinct,
                // disjoint from known" for an arbitrary known mask.
                #[allow(clippy::cast_possible_truncation)]
                let known = (lcg(&mut state) as u64) & (u64::MAX >> (64 - base));
                let mut seen = known;
                let mut ok = true;
                for &d in sq.iter().chain(cu.iter()) {
                    if seen & (1u64 << d) != 0 {
                        ok = false;
                    }
                    seen |= 1u64 << d;
                }
                assert_eq!(
                    survives(base, nmod, known),
                    ok,
                    "b{base} n={n} known={known:#x}"
                );
            }
        }
    }

    /// The stride walk enables the filter only from `b^5` up (see the module
    /// docs); every supported base's legal range starts above that, so the
    /// gate never switches the filter off in production.
    #[test_log::test]
    fn legal_ranges_start_above_the_gate() {
        for base in (2..=64u32).filter(|&b| supports(b, 3)) {
            let r = get_base_range_u128(base).unwrap().unwrap();
            assert!(
                r.start() >= u128::from(base).pow(5),
                "b{base}: range start {} is below b^5",
                r.start()
            );
        }
    }

    /// The word arithmetic at its extremes: `s` and `t` at 0, 1, 2, `B/2`,
    /// `B-2` and `B-1` (`B = b³`) for every specialized base, on numbers
    /// inside the legal range, against wide arithmetic.
    #[test_log::test]
    fn middle_digits_at_the_word_extremes() {
        for base in (2..=64u32).filter(|&b| supports(b, 3)) {
            let r = get_base_range_u128(base).unwrap().unwrap();
            let bk = u128::from(base).pow(3);
            let b6 = bk * bk;
            // First multiple of b^6 inside the range, with room to spare.
            let block = (r.start() / b6 + 1) * b6;
            assert!(block + b6 < r.end(), "b{base}: range too short");
            let edges = [0, 1, 2, bk / 2, bk - 2, bk - 1];
            for &s in &edges {
                for &t in &edges {
                    let n = block + s + bk * t;
                    #[allow(clippy::cast_possible_truncation)]
                    let (sq, cu) = middle_digits(base, (n % b6) as u64);
                    let (tsq, tcu) = true_digits(n, base);
                    assert_eq!(sq, tsq[3..6], "b{base} s={s} t={t}: square");
                    assert_eq!(cu, tcu[3..6], "b{base} s={s} t={t}: cube");
                }
            }
        }
    }

    #[test_log::test]
    fn unsupported_bases_never_filter() {
        assert!(!supports(10, 3));
        assert!(!supports(40, 2));
        assert!(!supports(65, 3));
        assert!(survives(10, 12345, u64::MAX));
        assert!(survives(65, 12345, u64::MAX));
    }

    #[test_log::test]
    fn masked_iteration_with_filter_matches_without_on_windows() {
        // End-to-end: the filtered stride iteration must return exactly the
        // same set as the plain seeded check. No base the filter supports
        // has a nice number anywhere near its legal range, so this runs just
        // above the filter's `b^5` gate instead: there `n²` and `n³` have
        // fewer than `b` digits between them, and the seeded check accepts
        // every candidate whose digits do not repeat, which is common. With
        // a fine floor (64) the certificates are rich. Every rejection by the
        // affine filter has to be a genuine repeat for the sets to agree.
        for base in [52u32, 64] {
            let table = StrideTable::new(base, 3);
            let start = u128::from(base).pow(5) * 7 / 3 + 12_345;
            let range = FieldSize::new(start, start + 4_000_000);
            let mut leaves = Vec::new();
            crate::msd_prefix_filter::get_valid_ranges_recursive_masked(
                range,
                &crate::msd_prefix_filter::MaskedRecursion {
                    base,
                    fixed_lsd_k: 3,
                    max_depth: crate::msd_prefix_filter::MSD_RECURSIVE_MAX_DEPTH,
                    min_range_size: 64,
                    subdivision_factor: 2,
                },
                0,
                0,
                &mut leaves,
            );
            assert!(!leaves.is_empty(), "b{base}: window fully rejected");
            let mut with = Vec::new();
            let mut without = Vec::new();
            for (leaf, hi) in leaves {
                with.extend(table.iterate_range_masked(&leaf, base, hi));
                without.extend(table.iterate_range_masked_without_affine(&leaf, base, hi));
            }
            assert!(!without.is_empty(), "b{base}: nothing to compare");
            assert_eq!(with, without, "b{base}");
        }
    }
}
