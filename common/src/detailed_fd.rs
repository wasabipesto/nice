//! Detailed mode by finite differences.
//!
//! Over a run of consecutive n, the square and cube follow from additions
//! alone, so this loop never divides:
//!
//! ```text
//! S  = n²   S  += dS    dS  = 2n + 1        dS  += 2
//! C  = n³   C  += dC    dC  = 3n² + 3n + 1  dC  += ddC
//!                       ddC = 6n + 6        ddC += 6
//! ```
//!
//! [`process_range_detailed`] squares and cubes every n and pulls each base-b
//! digit out with a multi-limb division, about 50 digit steps per number at
//! base 50. Here S, C and the three differences are kept as little-endian
//! base-b digit arrays (one digit per byte, with a length past which every
//! digit is zero), seeded once per range from the exact `U256` arithmetic and
//! then stepped to n + 1 by five array additions. Each addition handles eight
//! digits per native 64-bit add (see `add_word`), and every data-dependent
//! carry is arithmetic rather than a branch, so branch mispredictions stay
//! near zero; only rare events (a carry rippling past the word above the
//! addend, which needs eight digits of b − 1) branch. The digit-presence
//! mask is built from the `len` digits of S and C and `count_ones()` gives
//! `num_uniques`, the same mask and popcount as the division path, so
//! leading zeros never enter it and the results are identical, near-miss
//! list included.
//!
//! [`process_range_detailed_fd`] is the CPU detailed path: it runs this loop
//! for the bases the division path specialises at compile time and hands
//! every other base to [`process_range_detailed`], which stays the reference
//! the tests compare against.

// Hot-loop helpers: forced inlining is the point (each is called at a handful
// of sites inside the per-number loop).
#![allow(clippy::inline_always)]

use crate::client_process::process_range_detailed;
use crate::fixed_width::U256;
use crate::{FieldResults, FieldSize, NiceNumberSimple, UniquesDistributionSimple, number_stats};

/// Digit capacity of every array, a whole number of 64-bit words. The largest
/// value carried is n³: 38 digits at base 64 over its legal range. The slack
/// covers the word past the top that every addition touches and the
/// out-of-range test windows; an overflow panics on the bounds check, it
/// never wraps.
const CAP: usize = 72;

/// One in every byte of a word.
const LO: u64 = 0x0101_0101_0101_0101;

/// One word (8 digits, one per byte) of base-`BASE` addition: `a + r + cin`,
/// returning the sum and the carry out of the top byte.
///
/// Biased addition, as in BCD: add `256 - BASE` to every byte of the addend,
/// so a byte overflows into the next exactly when its digit sum reaches
/// `BASE`, and the native 64-bit add propagates all eight carries. The bytes
/// that did not carry out then take the bias back off. Needs every byte of
/// `a` and `r` below `BASE`. There is no per-digit compare, so nothing for
/// the compiler to turn into a branch.
#[inline(always)]
fn add_word<const BASE: u32>(a: u64, r: u64, cin: bool) -> (u64, bool) {
    let bias = 256 - u64::from(BASE);
    let rb = r + bias * LO; // per byte r + bias ≤ 255: no cross-byte carry
    let (t, c1) = a.overflowing_add(rb);
    let (t, c2) = t.overflowing_add(u64::from(cin));
    let cout = c1 | c2;
    // Bit 8j of t ^ a ^ rb is the carry into byte j; the carry out of byte j
    // is the carry into byte j + 1, and `cout` for the top byte.
    let carried = (((t ^ a ^ rb) >> 8) & LO) | (u64::from(cout) << 56);
    (t - (carried ^ LO) * bias, cout)
}

/// A non-negative integer as base-`BASE` digits, least significant first.
#[derive(Clone, Copy)]
struct Digits {
    d: [u8; CAP],
    /// Number of significant digits; `d[len..]` is all zero.
    len: usize,
}

impl Digits {
    #[allow(clippy::cast_possible_truncation)]
    fn from_u256<const BASE: u32>(mut x: U256) -> Self {
        let mut d = [0u8; CAP];
        let mut len = 0;
        while !x.is_zero() {
            d[len] = x.div_assign_rem_u32_const::<BASE>() as u8;
            len += 1;
        }
        Self { d, len }
    }

    fn from_u128<const BASE: u32>(x: u128) -> Self {
        Self::from_u256::<BASE>(U256::from_u128(x))
    }

    #[inline(always)]
    fn word(&self, i: usize) -> u64 {
        u64::from_le_bytes(self.d[8 * i..8 * i + 8].try_into().unwrap())
    }

    #[inline(always)]
    fn set_word(&mut self, i: usize, w: u64) {
        self.d[8 * i..8 * i + 8].copy_from_slice(&w.to_le_bytes());
    }

    /// Carry one into word `i` and up. Rare (the word below must have been
    /// eight digits of b − 1), so it stays out of line.
    #[inline(never)]
    fn ripple_words<const BASE: u32>(&mut self, mut i: usize) {
        loop {
            let (t, c) = add_word::<BASE>(self.word(i), 0, true);
            self.set_word(i, t);
            if !c {
                return;
            }
            i += 1;
        }
    }

    /// Update `len` after adding an operand of length `operand_len`: the sum
    /// has at most one digit more than the longer operand. Arithmetic, not a
    /// branch, since a new top digit is common in the short difference terms.
    #[inline(always)]
    fn fix_len(&mut self, operand_len: usize) {
        let len = self.len.max(operand_len);
        self.len = len + usize::from(self.d[len] != 0);
    }

    /// `self += rhs`, eight digits per 64-bit add.
    ///
    /// The carry out of the addend's top word lands in the next word whether
    /// or not it is set: it is a coin flip, and written as `if carry { .. }`
    /// it would compile to a branch that mispredicts. (An earlier digit-at-a-time
    /// version of this loop ran 3x slower for exactly that reason: LLVM turned
    /// its carry select into a branch, about 17 mispredictions per number.)
    /// Only a carry out of that word, which needs eight digits of b − 1,
    /// branches.
    #[inline(always)]
    fn add_assign<const BASE: u32>(&mut self, rhs: &Self) {
        let nw = rhs.len.div_ceil(8);
        let mut c = false;
        for i in 0..nw {
            let (t, co) = add_word::<BASE>(self.word(i), rhs.word(i), c);
            self.set_word(i, t);
            c = co;
        }
        let (t, co) = add_word::<BASE>(self.word(nw), 0, c);
        self.set_word(nw, t);
        if co {
            self.ripple_words::<BASE>(nw + 1);
        }
        self.fix_len(rhs.len);
    }

    /// `self += k` for a single digit `0 < k < BASE`.
    #[inline(always)]
    fn add_small<const BASE: u32>(&mut self, k: u8) {
        let (t, co) = add_word::<BASE>(self.word(0), u64::from(k), false);
        self.set_word(0, t);
        if co {
            self.ripple_words::<BASE>(1);
        }
        self.fix_len(1);
    }

    /// Bit `d` set for every significant digit `d`.
    #[inline(always)]
    fn mask(&self) -> u64 {
        // Two accumulators halve the OR dependency chain.
        let (pairs, rest) = self.d[..self.len].as_chunks::<2>();
        let (mut m0, mut m1) = (0u64, 0u64);
        for &[a, b] in pairs {
            m0 |= 1u64 << a;
            m1 |= 1u64 << b;
        }
        for &d in rest {
            m0 |= 1u64 << d;
        }
        m0 | m1
    }
}

/// The finite-difference state for n: digits of n², n³ and the three
/// differences that step them to n + 1.
struct State {
    s: Digits,
    ds: Digits,
    c: Digits,
    dc: Digits,
    ddc: Digits,
}

impl State {
    /// Seed the state at `n` with exact fixed-width arithmetic.
    fn new<const BASE: u32>(n: u128) -> Self {
        let square = U256::mul_u128_u128(n, n);
        let cube = square.mul_u128_truncating(n);
        // 3n² + 3n + 1 = 3·n(n+1) + 1, and 3·n(n+1) is even.
        let mut dc = U256::mul_u128_u128(n, n + 1).mul_u128_truncating(3);
        dc.limbs[0] |= 1;
        Self {
            s: Digits::from_u256::<BASE>(square),
            ds: Digits::from_u128::<BASE>(2 * n + 1),
            c: Digits::from_u256::<BASE>(cube),
            dc: Digits::from_u256::<BASE>(dc),
            ddc: Digits::from_u128::<BASE>(6 * (n + 1)),
        }
    }

    /// `num_uniques` for the current n.
    #[inline(always)]
    fn num_uniques(&self) -> u32 {
        (self.s.mask() | self.c.mask()).count_ones()
    }

    /// Advance from n to n + 1. S and C each read the old value of their
    /// difference, so the order matters.
    #[inline(always)]
    fn step<const BASE: u32>(&mut self) {
        self.s.add_assign::<BASE>(&self.ds);
        self.ds.add_small::<BASE>(2);
        self.c.add_assign::<BASE>(&self.dc);
        self.dc.add_assign::<BASE>(&self.ddc);
        self.ddc.add_small::<BASE>(6);
    }
}

/// Per-number `num_uniques` over `[start, end)` by finite differences,
/// passed to `f` in order.
#[inline(always)]
fn for_each_uniques<const BASE: u32>(start: u128, end: u128, mut f: impl FnMut(u128, u32)) {
    const { assert!(BASE >= 7 && BASE <= 64, "u64 mask, and 6 must be one digit") };
    if start >= end {
        return;
    }
    let mut st = State::new::<BASE>(start);
    for num in start..end {
        f(num, st.num_uniques());
        st.step::<BASE>();
    }
}

/// `process_range_detailed` for one base, by finite differences.
#[allow(clippy::cast_possible_truncation)]
fn run<const BASE: u32>(range: &FieldSize) -> FieldResults {
    let cutoff = number_stats::get_near_miss_cutoff(BASE);
    let mut nice_numbers: Vec<NiceNumberSimple> = Vec::new();
    let mut histogram = [0u128; 65];
    for_each_uniques::<BASE>(range.start(), range.end(), |number, num_uniques| {
        histogram[num_uniques as usize] += 1;
        if num_uniques > cutoff {
            nice_numbers.push(NiceNumberSimple {
                number,
                num_uniques,
            });
        }
    });
    let distribution = (1..=BASE)
        .map(|i| UniquesDistributionSimple {
            num_uniques: i,
            count: histogram[i as usize],
        })
        .collect();
    FieldResults {
        distribution,
        nice_numbers,
    }
}

type Kernel = fn(&FieldSize) -> FieldResults;

/// The finite-difference loop for `base`, if it has one: the bases
/// `get_num_unique_digits` specialises at compile time.
fn fd_kernel(base: u32) -> Option<Kernel> {
    Some(match base {
        40 => run::<40>,
        42 => run::<42>,
        43 => run::<43>,
        44 => run::<44>,
        45 => run::<45>,
        47 => run::<47>,
        48 => run::<48>,
        49 => run::<49>,
        50 => run::<50>,
        52 => run::<52>,
        53 => run::<53>,
        54 => run::<54>,
        55 => run::<55>,
        57 => run::<57>,
        58 => run::<58>,
        59 => run::<59>,
        60 => run::<60>,
        62 => run::<62>,
        64 => run::<64>,
        _ => return None,
    })
}

/// The CPU detailed path: same signature and output as
/// [`process_range_detailed`], by finite differences at bases 40–64 (those
/// with a legal range, except 63); every other base goes to
/// `process_range_detailed`.
///
/// **Range semantics**: half-open, [`range_start`, `range_end`).
#[must_use]
pub fn process_range_detailed_fd(range: &FieldSize, base: u32) -> FieldResults {
    match fd_kernel(base) {
        Some(kernel) => kernel(range),
        None => process_range_detailed(range, base),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base_range::{get_base_range_natural, get_base_range_u128};
    use crate::bench_defs::DETAILED_SCENARIOS;
    use crate::client_process::get_num_unique_digits;
    use malachite::base::num::arithmetic::traits::{CeilingRoot, Pow};
    use malachite::natural::Natural;

    /// splitmix64: a seeded, dependency-free PRNG for window starts.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        fn below(&mut self, n: u128) -> u128 {
            ((u128::from(self.next()) << 64) | u128::from(self.next())) % n
        }
    }

    /// Compare finite differences with `get_num_unique_digits` number by
    /// number over `[start, start + len)`.
    fn check_window<const BASE: u32>(start: u128, len: u128) {
        let mut expect = start;
        for_each_uniques::<BASE>(start, start + len, |n, u| {
            assert_eq!(n, expect);
            assert_eq!(
                u,
                get_num_unique_digits(n, BASE),
                "base {BASE}, n = {n} (window start {start})"
            );
            expect += 1;
        });
        assert_eq!(expect, start + len);
    }

    /// 20 seeded random windows of 10,000 across the legal range, plus both
    /// of its ends.
    fn random_windows<const BASE: u32>(seed: u64) {
        let range = get_base_range_u128(BASE).unwrap().unwrap();
        let len = 10_000;
        let mut rng = Rng(seed);
        for _ in 0..20 {
            let start = range.start() + rng.below(range.size() - len);
            check_window::<BASE>(start, len);
        }
        check_window::<BASE>(range.start(), len);
        check_window::<BASE>(range.end() - len, len);
    }

    /// Windows of 10,000 centred on every point from a factor `BASE` below
    /// the legal range up to its end where n² or n³ gains a digit: the
    /// smallest n with `n^root ≥ BASE^m`. (Past the end, n³ outgrows the
    /// reference's u128 path at base 40.)
    fn digit_gain_windows<const BASE: u32>() {
        let (lo, hi) = get_base_range_natural(BASE).unwrap();
        let lo = lo / Natural::from(BASE);
        let half = 5_000;
        let mut checked = 0;
        for (root, ms) in [(2u64, 0..140u64), (3, 0..200)] {
            for m in ms {
                let x = Natural::from(BASE).pow(m).ceiling_root(root);
                if x < lo || x > hi {
                    continue;
                }
                let x = u128::try_from(&x).unwrap();
                assert!(x > half);
                check_window::<BASE>(x - half, 2 * half);
                checked += 1;
            }
        }
        assert!(
            checked >= 3,
            "base {BASE}: only {checked} digit-gain windows"
        );
    }

    #[test]
    fn fd_matches_division_small_base_from_zero() {
        // From n = 0: the zero start, hundreds of digit gains in S, C and
        // the differences, and addends longer than the value they add to.
        check_window::<10>(0, 200_000);
        check_window::<7>(0, 50_000);
    }

    #[test]
    fn fd_matches_division_b40_random_windows() {
        random_windows::<40>(0x4040_4040);
    }

    #[test]
    fn fd_matches_division_b50_random_windows() {
        random_windows::<50>(0x5050_5050);
    }

    #[test]
    fn fd_matches_division_b60_random_windows() {
        random_windows::<60>(0x6060_6060);
    }

    #[test]
    fn fd_matches_division_other_bases_random_windows() {
        random_windows::<42>(42);
        random_windows::<47>(47);
        random_windows::<53>(53);
        random_windows::<59>(59);
        random_windows::<64>(64);
    }

    #[test]
    fn fd_matches_division_b40_digit_gains() {
        digit_gain_windows::<40>();
    }

    #[test]
    fn fd_matches_division_b50_digit_gains() {
        digit_gain_windows::<50>();
    }

    #[test]
    fn fd_matches_division_b60_digit_gains() {
        digit_gain_windows::<60>();
    }

    #[test]
    fn fd_field_results_match_detailed_benchmark_windows() {
        for def in DETAILED_SCENARIOS.iter().filter(|d| !d.single_thread) {
            let start = def.resolved_start();
            let range = FieldSize::new(start, start + def.window_cpu);
            let old = process_range_detailed(&range, def.base);
            let new = process_range_detailed_fd(&range, def.base);
            assert_eq!(old, new, "{}", def.key);
            let total: u128 = new.distribution.iter().map(|d| d.count).sum();
            assert_eq!(total, def.window_cpu, "{}", def.key);
        }
    }

    #[test]
    fn fd_field_results_match_small_bases_full_range() {
        // The benchmark windows hold no near-misses, so the list itself is
        // checked here: the whole range of every base from 10 to 24 that has
        // one, through the same generic loop.
        fn check<const B: u32>() -> usize {
            let range = get_base_range_u128(B).unwrap().unwrap();
            let new = run::<B>(&range);
            assert_eq!(new, process_range_detailed(&range, B), "base {B}");
            new.nice_numbers.len()
        }
        let near_misses = check::<10>()
            + check::<12>()
            + check::<13>()
            + check::<14>()
            + check::<15>()
            + check::<17>()
            + check::<18>()
            + check::<19>()
            + check::<20>()
            + check::<22>()
            + check::<23>()
            + check::<24>();
        // 27 at the current near-miss cutoff.
        assert_eq!(near_misses, 27);
    }

    #[test]
    fn dispatch_uses_finite_differences_where_supported() {
        // The detailed benchmark bases take the new loop.
        assert!(fd_kernel(40).is_some());
        assert!(fd_kernel(50).is_some());
        // So does every base from 40 to 64 with a legal range, except 63,
        // which the division path does not specialise either.
        for base in 40..=64 {
            let has_range = get_base_range_u128(base).unwrap().is_some();
            assert_eq!(
                fd_kernel(base).is_some(),
                has_range && base != 63,
                "base {base}"
            );
        }
        // Other bases stay on the division loop, with its results.
        assert!(fd_kernel(10).is_none());
        let range = get_base_range_u128(10).unwrap().unwrap();
        assert_eq!(
            process_range_detailed_fd(&range, 10),
            process_range_detailed(&range, 10)
        );
    }

    /// Single-thread timing of the division loop against finite differences
    /// on the same 1,000,000-number window at the start of each base's range,
    /// interleaved over three rounds. Prints ns per number.
    ///
    /// `cargo test --release -p nice_common --lib fd_timing -- --ignored --nocapture --test-threads 1`
    #[test]
    #[ignore = "timing, run by hand"]
    #[allow(clippy::cast_precision_loss)]
    fn fd_timing() {
        use std::hint::black_box;
        use std::time::Instant;
        let window: u128 = 1_000_000;
        let fmt = |v: &[f64]| {
            v.iter()
                .map(|x| format!("{x:.2}"))
                .collect::<Vec<_>>()
                .join(" ")
        };
        let med = |v: &[f64]| {
            let mut s = v.to_vec();
            s.sort_by(f64::total_cmp);
            s[s.len() / 2]
        };
        for base in [40u32, 50, 60] {
            let start = get_base_range_u128(base).unwrap().unwrap().start();
            let range = FieldSize::new(start, start + window);
            let (mut old, mut new) = (Vec::new(), Vec::new());
            for _ in 0..3 {
                let t = Instant::now();
                let a = black_box(process_range_detailed(black_box(&range), base));
                old.push(t.elapsed().as_nanos() as f64 / window as f64);
                let t = Instant::now();
                let b = black_box(process_range_detailed_fd(black_box(&range), base));
                new.push(t.elapsed().as_nanos() as f64 / window as f64);
                assert_eq!(a, b);
            }
            eprintln!(
                "TIMING b{base}: division [{}] med {:.2} | finite differences [{}] med {:.2} | {:.2}x",
                fmt(&old),
                med(&old),
                fmt(&new),
                med(&new),
                med(&old) / med(&new),
            );
        }
    }
}
