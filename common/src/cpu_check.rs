//! The CPU join's full check of a survivor, from the digits the join has
//! already proved ([`crate::cpu_join`]).
//!
//! A survivor `n` of a top `P` (`n = P·b^f0 + r0`) has passed the AND and
//! the prefilter, so the digits of `n²` and `n³` at positions `0..k2` are
//! known, and distinct: the bottom's `2k` (its mask) and the prefilter's
//! four. When the top's certificate sits at `k2` or above (its seed flag),
//! the digits from the certificate's lowest common position of each power
//! up are known too, the same for every `n` of the block, and distinct from
//! the rest. So `n` is nice exactly when the digits in between, the middle
//! of each power, are distinct and new against those: the check seeds its
//! mask with the known digits and converts only the middle, stopping at the
//! first repeat.
//!
//! It converts them without dividing a long number. `n` is written in
//! base-b² limbs (two digits each) from `P` and `r0`: `n = H·b^e + L` for
//! an even `e` (`min(2·nl, 10)` at bases 39-64) with `L = n mod b^e` below `2^63` and
//! `H` below `2^31`, both from one quotient of `P` by `b^(e − f0)`
//! ([`Magic64`]); `L`'s limbs come from its quotients by `b^2i`, each
//! taken from `L` directly, and `H`'s from quotients by `b²`. `n²` and
//! `n³` are schoolbook columns of `u32` products from the low end, each
//! below `2^31`, normalised to limbs by a multiply-high quotient by `b²`
//! that is exact below `2^31` ([`Magic`]), and a limb is split into its
//! digits by one by `b`. The columns below `k2` only carry. Fields whose
//! `n` does not fit those bounds (none in production) take the same digits
//! by long division of a 256-bit `n²` and `n³`.

// The quotients and the column steps are forced inline: they are the
// check's per-digit work, and the column form is written out per limb.
#![allow(clippy::inline_always)]

use crate::fixed_width::U256;
use anyhow::{Result, ensure};

/// `⌊x / d⌋ = (x·m) >> s` for every `x < 2^31`: for `d` a power of two `m =
/// 1`; otherwise `s = 32 + ⌊log2 d⌋` and `m = ⌈2^s / d⌉ < 2^32`. With `e =
/// m·d − 2^s` (`0 < e < d`), `x·m / 2^s = x/d + x·e / (d·2^s)`, and the
/// second term is below `1/d` when `x·e < 2^s`, which `x < 2^31` and `e <
/// d < 2^(s − 31)` give: the quotient is exact (the tests' `bound`).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Magic {
    d: u32,
    m: u64,
    s: u32,
}

impl Magic {
    /// # Panics
    /// If `d` is 0.
    #[must_use]
    pub(crate) const fn new(d: u32) -> Self {
        assert!(d > 0, "Magic: a divisor of 0");
        if d.is_power_of_two() {
            return Self {
                d,
                m: 1,
                s: d.trailing_zeros(),
            };
        }
        let s = 32 + d.ilog2();
        Self {
            d,
            m: (1u64 << s).div_ceil(d as u64),
            s,
        }
    }

    /// `⌊x / d⌋`, for `x < 2^31`.
    // The quotient is below x < 2^31.
    #[allow(clippy::cast_possible_truncation)]
    #[inline(always)]
    pub(crate) fn div(self, x: u32) -> u32 {
        debug_assert!(x < 1 << 31, "Magic::div({x}): outside its range");
        ((u64::from(x) * self.m) >> self.s) as u32
    }

    /// `(⌊x / d⌋, x mod d)`, for `x < 2^31`.
    #[inline(always)]
    pub(crate) fn divrem(self, x: u32) -> (u32, u32) {
        let q = self.div(x);
        (q, x - q * self.d)
    }

    /// The first `x` at which the sufficient condition `x·e < 2^s` fails
    /// (`u64::MAX` for a power of two, where the shift is exact).
    #[cfg(test)]
    #[must_use]
    pub(crate) fn bound(self) -> u64 {
        if self.d.is_power_of_two() {
            return u64::MAX;
        }
        let e = self.m * u64::from(self.d) - (1u64 << self.s);
        (1u64 << self.s).div_ceil(e)
    }
}

/// [`Magic`] for `u64` dividends below `2^63`: `⌊x / d⌋ = ⌊x·m / 2^s⌋`
/// with `s = 64 + ⌊log2 d⌋` and `m = ⌈2^s / d⌉ < 2^64` (`m = 2^(64 −
/// log2 d)`, `s = 64` for a power of two `d >= 2`); exact for `x·e < 2^s`,
/// `e = m·d − 2^s < d`, so for every `x < 2^63`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Magic64 {
    d: u64,
    m: u64,
    s: u32,
}

impl Magic64 {
    /// # Panics
    /// If `d` is below 2 or not below `2^63`.
    #[must_use]
    pub(crate) fn new(d: u64) -> Self {
        assert!(d > 1 && d < 1 << 63, "Magic64: a divisor of {d}");
        if d.is_power_of_two() {
            return Self {
                d,
                m: 1 << (64 - d.trailing_zeros()),
                s: 64,
            };
        }
        let s = 64 + d.ilog2();
        let m = (1u128 << s).div_ceil(u128::from(d));
        Self {
            d,
            m: u64::try_from(m).expect("m < 2^64 for d not a power of 2"),
            s,
        }
    }

    /// `(⌊x / d⌋, x mod d)`, for `x < 2^63`.
    // The high word of the product.
    #[allow(clippy::cast_possible_truncation)]
    #[inline(always)]
    pub(crate) fn divrem(self, x: u64) -> (u64, u64) {
        debug_assert!(x < 1 << 63, "Magic64::divrem({x}): outside its range");
        let q = (((u128::from(x) * u128::from(self.m)) >> 64) as u64) >> (self.s - 64);
        (q, x - q * self.d)
    }

    /// As [`Magic::bound`].
    #[cfg(test)]
    #[must_use]
    pub(crate) fn bound(self) -> u128 {
        if self.d.is_power_of_two() {
            return u128::MAX;
        }
        let e = u128::from(self.m) * u128::from(self.d) - (1u128 << self.s);
        (1u128 << self.s).div_ceil(e)
    }
}

/// The most limbs of `n` the column form takes (`n` of up to 16 digits),
/// and of `n²` and `n³`.
const MAX_NL: usize = 8;
const SQ_LIMBS: usize = 2 * MAX_NL;

/// The digit mask a check runs against, whether a digit it took repeated,
/// and in count mode (`Check::count`) every digit it took.
struct Scan<const COUNT: bool> {
    seen: u64,
    repeat: bool,
    /// Count mode: each power's digits by position (`u8::MAX` where none
    /// was taken).
    trace: [[u8; 48]; 2],
}

impl<const COUNT: bool> Scan<COUNT> {
    fn new(seen: u64) -> Self {
        Self {
            seen,
            repeat: false,
            trace: [[u8::MAX; 48]; 2],
        }
    }

    /// Take digit `d` of power `j` (0: `n²`, 1: `n³`) at `pos` if `on`,
    /// without a branch: a repeat sets `repeat`, which the caller tests
    /// every few digits.
    // A digit is below b <= 64 and a position below 48.
    #[allow(clippy::cast_possible_truncation)]
    #[inline(always)]
    fn take(&mut self, j: usize, pos: u32, d: u32, on: bool) {
        let bit = u64::from(on) << d;
        self.repeat |= self.seen & bit != 0;
        self.seen |= bit;
        if COUNT && on {
            self.trace[j][pos as usize] = d as u8;
        }
    }
}

/// What a check in count mode converted ([`Check::count`]): the mask (the
/// seed and every middle digit), whether a middle digit repeated, and the
/// middle digits by position.
#[cfg(test)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Count {
    pub(crate) mask: u64,
    pub(crate) repeat: bool,
    pub(crate) trace: [[u8; 48]; 2],
}

/// One field's check: its shape, and the column form's constants when the
/// field takes it.
pub(crate) struct Check {
    b: u32,
    /// The known low digits of both powers.
    k2: u32,
    /// `b^f0`, for `n` itself on the long-division path.
    w: u128,
    /// Limbs of `n` (the column form's `NL`), or 0: the long division.
    nl: u32,
    by_b: Magic,
    by_bb: Magic,
    /// `n = H·b^e + L` for an even `e` with `L < b^e < 2^63` and `H <
    /// 2^31`: `P = H·b^(e − f0) + W` by `by_pe`, and `L = W·b^f0 + r0`;
    /// `L` holds `n`'s limbs below `eh = e/2`, `H` the rest.
    by_pe: Magic64,
    bf0: u64,
    eh: u32,
    /// By `b^(2i)`, for `L`'s limb `i` (`0 < i < eh`).
    by_l: [Magic64; MAX_NL],
}

impl Check {
    /// The check of a field of `n` with `l` digits whose squares and cubes
    /// have `s2` and `s3`, at base `b`, for survivors `n = P·b^f0 + r0` with
    /// `P <= p_max` whose digits below `k2` are known.
    ///
    /// # Errors
    /// A shape no check here takes: a base outside 3-64, digit counts no
    /// `n` of `l` digits has, a cube that could reach `2^256`, or `k2`
    /// outside `f0 + 1..=s2`.
    // b, l, s2, s3, f0, k2 are the join's names.
    #[allow(clippy::many_single_char_names)]
    pub(crate) fn new(
        b: u32,
        l: u32,
        s2: u32,
        s3: u32,
        f0: u32,
        k2: u32,
        p_max: u128,
    ) -> Result<Self> {
        ensure!((3..=64).contains(&b), "check: base {b} outside 3-64");
        ensure!(
            l >= 1 && (2 * l - 1..=2 * l).contains(&s2) && (3 * l - 2..=3 * l).contains(&s3),
            "check: no n of {l} digits has a square of {s2} digits and a cube of {s3}"
        );
        // n³ < b^s3 <= 2^256, as the long division needs.
        ensure!(
            s3 <= 48 && s3 * b.next_power_of_two().ilog2() <= 256,
            "check: a cube of {s3} digits at base {b}"
        );
        ensure!(f0 < k2 && k2 <= s2, "check: f0 {f0}, k2 {k2}, s2 {s2}");
        let bb = u128::from(b);
        // The column form: n of at most MAX_NL limbs, P below 2^63, and an
        // even e > f0 with b^e < 2^63 and H = ⌊n / b^e⌋ below 2^31. Its
        // columns are then below 2^31: a column has at most nl products of
        // limbs below b², and a carry below nl·b² (by induction), so it is
        // below nl·(b⁴ + b²) <= 8·(2^24 + 2^12) at b <= 64.
        let nl = l.div_ceil(2);
        let mut e = 2 * nl;
        while e > 0 && bb.pow(e) >= 1 << 63 {
            e -= 2;
        }
        let column = u128::from(nl) * (bb.pow(4) + bb * bb);
        let narrow = nl as usize <= MAX_NL
            && p_max < 1 << 63
            && e > f0
            && bb.pow(l.saturating_sub(e)) < 1 << 31
            && column < 1 << 31;
        // A divisor of the column form (unused unless it takes the field).
        let by = |i: u32| Magic64::new(u64::try_from(bb.pow(i)).unwrap_or(2).max(2));
        let mut by_l = [Magic64::new(2); MAX_NL];
        for (i, m) in (0u32..).zip(&mut by_l) {
            if i > 0 && i < e / 2 {
                *m = by(2 * i);
            }
        }
        Ok(Self {
            b,
            k2,
            w: bb.pow(f0),
            nl: if narrow { nl } else { 0 },
            by_b: Magic::new(b),
            by_bb: Magic::new(b * b),
            by_pe: by(e.saturating_sub(f0)),
            bf0: u64::try_from(bb.pow(f0))?,
            eh: e / 2,
            by_l,
        })
    }

    /// Whether `n = P·b^f0 + r0` is nice, for a survivor whose digits
    /// below `k2` are `low` (distinct) and whose top certifies `top`
    /// (distinct, and disjoint from `low`): the digits of `n²` at
    /// positions `k2..ends[0]` and of `n³` at `k2..ends[1]` distinct and
    /// new against `low | top`; `top` holds the digits at and above
    /// `ends` (none where `ends` is `(s2, s3)`).
    #[inline(always)]
    pub(crate) fn nice(&self, p: u128, r0: u32, low: u64, top: u64, ends: [u32; 2]) -> bool {
        let seen = low | top;
        // P < 2^63 wherever `nl` is set.
        #[allow(clippy::cast_possible_truncation)]
        let p64 = p as u64;
        match self.nl {
            1 => self.columns::<1, false>(p64, r0, seen, ends).0,
            2 => self.columns::<2, false>(p64, r0, seen, ends).0,
            3 => self.columns::<3, false>(p64, r0, seen, ends).0,
            4 => self.columns::<4, false>(p64, r0, seen, ends).0,
            5 => self.columns::<5, false>(p64, r0, seen, ends).0,
            6 => self.columns::<6, false>(p64, r0, seen, ends).0,
            7 => self.columns::<7, false>(p64, r0, seen, ends).0,
            8 => self.columns::<8, false>(p64, r0, seen, ends).0,
            _ => {
                self.long::<false>(p * self.w + u128::from(r0), seen, ends)
                    .0
            }
        }
    }

    /// [`Self::nice`] in count mode: every middle digit converted, none
    /// skipped at a repeat (the path the field takes, or the long division
    /// if `long`).
    #[cfg(test)]
    pub(crate) fn count(&self, p: u128, r0: u32, seen: u64, ends: [u32; 2], long: bool) -> Count {
        #[allow(clippy::cast_possible_truncation)]
        let p64 = p as u64;
        let scan = match if long { 0 } else { self.nl } {
            1 => self.columns::<1, true>(p64, r0, seen, ends).1,
            2 => self.columns::<2, true>(p64, r0, seen, ends).1,
            3 => self.columns::<3, true>(p64, r0, seen, ends).1,
            4 => self.columns::<4, true>(p64, r0, seen, ends).1,
            5 => self.columns::<5, true>(p64, r0, seen, ends).1,
            6 => self.columns::<6, true>(p64, r0, seen, ends).1,
            7 => self.columns::<7, true>(p64, r0, seen, ends).1,
            8 => self.columns::<8, true>(p64, r0, seen, ends).1,
            _ => self.long::<true>(p * self.w + u128::from(r0), seen, ends).1,
        };
        Count {
            mask: scan.seen,
            repeat: scan.repeat,
            trace: scan.trace,
        }
    }

    /// Whether the field's survivors take the column form.
    #[cfg(test)]
    pub(crate) fn columnar(&self) -> bool {
        self.nl > 0
    }

    /// `n`'s base-b² limbs from `P` and `r0`: `L = n mod b^e` (below
    /// 2^63) and `H = ⌊n / b^e⌋` (below 2^31) from `P`'s quotient by
    /// `b^(e − f0)`; limb `i < e/2` is `⌊L / b^(2i)⌋ − ⌊L / b^(2i + 2)⌋·b²`,
    /// each quotient taken from `L` directly, and `H`'s by quotients by `b²`.
    // Limbs are below b²; H below 2^31 (`Self::new`). P, H, W, q as above.
    #[allow(clippy::cast_possible_truncation, clippy::many_single_char_names)]
    #[inline(always)]
    pub(crate) fn limbs<const NL: usize>(&self, p: u64, r0: u32) -> [u32; NL] {
        let (h, w) = self.by_pe.divrem(p);
        let lo = w * self.bf0 + u64::from(r0);
        let eh = self.eh as usize;
        let bb = u64::from(self.b * self.b);
        let mut q = [0u64; MAX_NL + 1];
        q[0] = lo;
        // An index loop: the iterator form's `skip` is an out-of-line call.
        #[allow(clippy::needless_range_loop)]
        for i in 1..NL {
            if i < eh {
                q[i] = self.by_l[i].divrem(lo).0;
            }
        }
        let mut nn = [0u32; NL];
        let mut h = h as u32;
        for (i, limb) in nn.iter_mut().enumerate() {
            *limb = match (i + 1).cmp(&eh) {
                std::cmp::Ordering::Less => (q[i] - q[i + 1] * bb) as u32,
                std::cmp::Ordering::Equal => q[i] as u32,
                std::cmp::Ordering::Greater => {
                    let (qh, r) = self.by_bb.divrem(h);
                    h = qh;
                    r
                }
            };
        }
        nn
    }

    /// The column form: `n²`'s limbs from the low end, its middle digits
    /// against the mask, then (if they pass) `n³ = n²·n`'s the same way.
    /// Two digits a column are taken without a branch, and the scan stops
    /// at a repeat every other column (most survivors are out after the
    /// first four middle digits).
    #[allow(clippy::many_single_char_names)] // the arithmetic's names
    #[inline(always)]
    fn columns<const NL: usize, const COUNT: bool>(
        &self,
        p: u64,
        r0: u32,
        seen: u64,
        ends: [u32; 2],
    ) -> (bool, Scan<COUNT>) {
        let nn = self.limbs::<NL>(p, r0);
        let mut sc = Scan::<COUNT>::new(seen);
        let k2 = self.k2;
        let mut sq = [0u32; SQ_LIMBS];
        let mut carry = 0u32;
        // Column `j` of n², written out for every `j` (the loop's indices
        // are then constants): twice the products i < j − i, the square of
        // limb j/2, and the carry; its digits are 2j and 2j + 1. n² <
        // (b²)^(2·nl) has 2·nl columns (the top one 0 for an odd l).
        macro_rules! column {
            ($($j:literal)*) => {$(
                if $j < 2 * NL {
                    let mut x = 0u32;
                    for i in ($j as usize).saturating_sub(NL - 1)..($j as usize).div_ceil(2) {
                        x += nn[i] * nn[$j - i];
                    }
                    x = 2 * x + carry;
                    if $j % 2 == 0 && $j / 2 < NL {
                        x += nn[$j / 2] * nn[$j / 2];
                    }
                    let (q, limb) = self.by_bb.divrem(x);
                    carry = q;
                    sq[$j] = limb;
                    let pos: u32 = 2 * $j;
                    if pos + 2 > k2 && pos < ends[0] {
                        let (hi, lo) = self.by_b.divrem(limb);
                        sc.take(0, pos, lo, pos >= k2);
                        sc.take(0, pos + 1, hi, pos + 1 < ends[0]);
                        if !COUNT && $j % 2 == 1 && sc.repeat {
                            return (false, sc);
                        }
                    }
                }
            )*};
        }
        column!(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15);
        // The top column's carry is 0 (n² < (b²)^(2·nl)).
        debug_assert_eq!(carry, 0);
        if !COUNT && sc.repeat {
            return (false, sc);
        }
        let mut carry = 0u32;
        for j in 0..3 * NL {
            #[allow(clippy::cast_possible_truncation)]
            let pos = 2 * j as u32;
            if pos >= ends[1] {
                break;
            }
            let mut x = carry;
            for (t, &a) in nn.iter().enumerate() {
                if t <= j && j - t < 2 * NL {
                    x += sq[j - t] * a;
                }
            }
            let (q, limb) = self.by_bb.divrem(x);
            carry = q;
            if pos + 2 > k2 {
                let (hi, lo) = self.by_b.divrem(limb);
                sc.take(1, pos, lo, pos >= k2);
                sc.take(1, pos + 1, hi, pos + 1 < ends[1]);
                if !COUNT && j % 2 == 1 && sc.repeat {
                    return (false, sc);
                }
            }
        }
        (!sc.repeat, sc)
    }

    /// The same digits by long division of `n²` and `n³` (any field).
    fn long<const COUNT: bool>(&self, n: u128, seen: u64, ends: [u32; 2]) -> (bool, Scan<COUNT>) {
        let mut sc = Scan::<COUNT>::new(seen);
        let sq = U256::mul_u128_u128(n, n);
        for (j, (mut x, end)) in [(sq, ends[0]), (sq.mul_u128_truncating(n), ends[1])]
            .into_iter()
            .enumerate()
        {
            for pos in 0..end {
                let d = x.div_assign_rem_u32(self.b);
                sc.take(j, pos, d, pos >= self.k2);
                if !COUNT && sc.repeat {
                    return (false, sc);
                }
            }
        }
        (!sc.repeat, sc)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base_range::get_base_range_u128;
    use crate::client_process::{get_is_nice, get_num_unique_digits};
    use malachite::base::num::arithmetic::traits::Pow;
    use malachite::base::num::conversion::traits::Digits;
    use malachite::natural::Natural;

    /// The digits of `n²` and `n³`, low first (malachite).
    fn digits(n: u128, b: u32) -> (Vec<u32>, Vec<u32>) {
        let nat = Natural::from(n);
        (
            (&nat * &nat).to_digits_asc(&b),
            (&nat).pow(3).to_digits_asc(&b),
        )
    }

    fn bits(d: &[u32]) -> u64 {
        d.iter().fold(0, |a, &x| a | 1u64 << x)
    }

    fn distinct(d: &[u32]) -> bool {
        let mut seen = 0u64;
        d.iter().all(|&x| {
            let new = seen & 1 << x == 0;
            seen |= 1 << x;
            new
        })
    }

    fn ndigits(mut x: u128, b: u32) -> u32 {
        let mut n = 0;
        while x > 0 {
            x /= u128::from(b);
            n += 1;
        }
        n
    }

    fn xorshift(x: &mut u64) -> u64 {
        *x ^= *x << 13;
        *x ^= *x >> 7;
        *x ^= *x << 17;
        *x
    }

    /// The divisors of [`Magic`]: `b` and `b²` at every base 3-64.
    fn divisors() -> Vec<u32> {
        (3..=64u32).flat_map(|b| [b, b * b]).collect()
    }

    /// The quotients by `b` and `b²` are exact on every input the check
    /// gives them: the bound of [`Magic`] is at least 2^31 for every
    /// divisor, and the quotient is exact at the worst element of every
    /// block of `d` below the inputs (`(x·m) >> s` is monotone in `x` and
    /// never below `⌊x/d⌋`, as `m·d >= 2^s`, so if it is exact at `q·d + d
    /// − 1` it is exact on all of `[q·d, q·d + d)`): by `b²`, every `x <
    /// 2^31` (columns, and the limbs of `H`); by `b`, every limb (below
    /// `b²`). [`Magic64`]'s bound is at least 2^63 for every power of every
    /// base below 2^63, and its quotients match `/` at the ends of blocks
    /// and at random.
    #[test]
    fn magic_quotients_are_exact_on_the_checks_inputs() {
        for d in divisors() {
            assert!(Magic::new(d).bound() >= 1 << 31, "d = {d}");
        }
        let check = |d: u32, below: u64| {
            let m = Magic::new(d);
            let d64 = u64::from(d);
            let mut q = 0u64;
            while q * d64 < below {
                let x = u32::try_from((q * d64 + d64 - 1).min(below - 1)).unwrap();
                assert_eq!(m.divrem(x), (x / d, x % d), "d = {d}, x = {x}");
                q += 1;
            }
        };
        for b in 3..=64u32 {
            check(b * b, 1 << 31);
            check(b, u64::from(b * b));
        }
        let mut r = 0x9E37_79B9_7F4A_7C15u64;
        for b in 3..=64u64 {
            let mut d = b;
            while d < 1 << 63 {
                let m = Magic64::new(d);
                assert!(m.bound() >= 1 << 63, "b{b}: d = {d}");
                let mut xs = vec![0, 1, d - 1, d, d + 1, (1 << 63) - 1];
                for _ in 0..2000 {
                    let x = xorshift(&mut r) >> (1 + xorshift(&mut r) % 63);
                    xs.extend([x, x / d * d, (x / d * d).saturating_sub(1)]);
                }
                for x in xs {
                    let x = x.min((1 << 63) - 1);
                    assert_eq!(m.divrem(x), (x / d, x % d), "d = {d}, x = {x}");
                }
                let Some(next) = d.checked_mul(b) else { break };
                d = next;
            }
        }
    }

    /// Opt-in (`NICE_TEST_CPU_CHECK_EXHAUSTIVE=1`; minutes on every core):
    /// [`Magic`] against `/` and `%` for every `x < 2^31` and every divisor
    /// `b`, `b²` (bases 3-64) that is not a power of two.
    #[test]
    #[ignore = "exhaustive, minutes on every core; NICE_TEST_CPU_CHECK_EXHAUSTIVE=1"]
    fn magic_quotients_are_exact_everywhere() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        if std::env::var("NICE_TEST_CPU_CHECK_EXHAUSTIVE").is_err() {
            eprintln!("skipping: set NICE_TEST_CPU_CHECK_EXHAUSTIVE=1 to run every x below 2^31");
            return;
        }
        let ds: Vec<u32> = divisors()
            .into_iter()
            .filter(|d| !d.is_power_of_two())
            .collect();
        let threads = std::thread::available_parallelism().map_or(4, usize::from);
        let next = AtomicUsize::new(0);
        std::thread::scope(|sc| {
            for _ in 0..threads {
                sc.spawn(|| {
                    while let Some(&d) = ds.get(next.fetch_add(1, Ordering::Relaxed)) {
                        let m = Magic::new(d);
                        let mut bad = 0u32;
                        for x in 0..1u32 << 31 {
                            bad += u32::from(m.div(x) != x / d);
                        }
                        assert_eq!(bad, 0, "d = {d}");
                    }
                });
            }
        });
        eprintln!("{} divisors exact on every x below 2^31", ds.len());
    }

    /// The join's shapes for `n`: `(f0, k2)` of the production parameters
    /// and of the test windows.
    const SHAPES: &[(u32, u32)] = &[
        (3, 8),
        (0, 3),
        (1, 3),
        (1, 4),
        (2, 4),
        (1, 6),
        (2, 6),
        (3, 6),
        (2, 7),
        (3, 7),
    ];

    /// The check of `n`'s field in shape `(f0, k2)`, if it can take one
    /// (`k2 <= s2`, and `P < 2^63` for the column form).
    fn check_for(n: u128, b: u32, f0: u32, k2: u32) -> Option<(Check, u32, u32)> {
        let l = ndigits(n, b);
        let (d2, d3) = digits(n, b);
        let (s2, s3) = (
            u32::try_from(d2.len()).unwrap(),
            u32::try_from(d3.len()).unwrap(),
        );
        let p = n / u128::from(b).pow(f0);
        Check::new(b, l, s2, s3, f0, k2, p)
            .ok()
            .map(|c| (c, s2, s3))
    }

    /// Adversarial `n` of `l` digits: every digit `b − 1` (the largest
    /// columns), the smallest, 0 and `b − 1` alternating either way, `b −
    /// 1` but one digit, `b^i − 1` above the smallest, and multiples of
    /// `b^f` at the parts' edges.
    fn adversarial(b: u32, l: u32) -> Vec<u128> {
        let bb = u128::from(b);
        let top = bb.pow(l) - 1;
        let low = bb.pow(l - 1);
        let alt = |first: u32| -> u128 {
            (0..l).fold(0u128, |a, i| {
                let d = if (i + first).is_multiple_of(2) {
                    b - 1
                } else {
                    0
                };
                a * bb + u128::from(if i == 0 && d == 0 { 1 } else { d })
            })
        };
        let mut v = vec![top, top - 1, low, low + 1, alt(0), alt(1)];
        for i in 0..l {
            v.push(top - u128::from(b - 1) * bb.pow(i));
            v.push(low + bb.pow(i) - 1);
        }
        for f in [2u32, 3, 8, 10] {
            if f < l {
                let w = bb.pow(f);
                v.extend([top / w * w, low / w * w + w - 1, low / w * w + w]);
            }
        }
        v.retain(|&n| (low..=top).contains(&n));
        v
    }

    /// The column form's digits are `n²`'s and `n³`'s: in count mode, with
    /// nothing seeded and the scan to the top of both powers, every digit
    /// from `k2` up equals malachite's, at every base 3-64, every length up
    /// to the column form's (`n < 2^96`), every shape, on adversarial and
    /// random `n`; the long division takes the same digits; and a scan that
    /// ends lower (random ends) takes exactly the digits below its ends.
    #[test]
    fn the_columns_take_the_digits_of_the_square_and_the_cube() {
        let mut r = 0x2545_F491_4F6C_DD1Du64;
        let mut runs = 0u64;
        for b in 3..=64u32 {
            let bb = u128::from(b);
            for l in 2..=16u32 {
                if bb.pow(l - 1) >= 1 << 85 {
                    break;
                }
                let (lo, hi) = (bb.pow(l - 1), (bb.pow(l) - 1).min((1 << 85) - 1));
                let mut ns = adversarial(b, l);
                for _ in 0..6 {
                    let x = (u128::from(xorshift(&mut r)) << 64) | u128::from(xorshift(&mut r));
                    ns.push(lo + x % (hi - lo + 1));
                }
                for n in ns {
                    let (d2, d3) = digits(n, b);
                    for &(f0, k2) in SHAPES {
                        let Some((check, s2, s3)) = check_for(n, b, f0, k2) else {
                            continue;
                        };
                        let w = bb.pow(f0);
                        let (p, r0) = (n / w, u32::try_from(n % w).unwrap());
                        let ends = [s2, s3];
                        let got = check.count(p, r0, 0, ends, false);
                        for (j, d) in [&d2, &d3].into_iter().enumerate() {
                            for (pos, &want) in d.iter().enumerate() {
                                let g = u32::from(got.trace[j][pos]);
                                let want = if pos < k2 as usize { 255 } else { want };
                                assert_eq!(
                                    g,
                                    want,
                                    "b{b} n = {n} f0 {f0} k2 {k2}: power {} digit {pos}",
                                    j + 2
                                );
                            }
                        }
                        assert_eq!(
                            got.mask,
                            bits(&d2[k2 as usize..]) | bits(&d3[k2 as usize..])
                        );
                        assert_eq!(
                            check.count(p, r0, 0, ends, true),
                            got,
                            "b{b} n = {n}: long division"
                        );
                        // Random ends: the digits below them, none above.
                        let e = [
                            k2 + u32::try_from(xorshift(&mut r) % u64::from(s2 - k2 + 1)).unwrap(),
                            k2 + u32::try_from(xorshift(&mut r) % u64::from(s3 - k2 + 1)).unwrap(),
                        ];
                        let cut = check.count(p, r0, 0, e, false);
                        for j in 0..2 {
                            for pos in 0..48 {
                                let want = if pos < e[j] as usize {
                                    got.trace[j][pos]
                                } else {
                                    255
                                };
                                assert_eq!(cut.trace[j][pos], want, "b{b} n = {n} ends {e:?}");
                            }
                        }
                        runs += u64::from(check.columnar());
                    }
                }
            }
        }
        eprintln!("{runs} n and shapes through the column form, each also by long division");
        assert!(runs > 30_000, "{runs} column-form runs");
    }

    /// The verdict path (`nice`, which stops at a repeat) agrees with
    /// count mode's `!repeat` on any mask, including masks that miss every
    /// middle digit, so that its `true` is taken at every limb count (no
    /// nice `n` is known past base 10, so the other tests only see it at
    /// 69, one limb).
    #[test]
    fn the_verdict_agrees_with_count_mode_on_any_mask() {
        let mut r = 0x1234_5678_9ABC_DEF1u64;
        let (mut runs, mut yes) = (0u64, [0u64; MAX_NL + 1]);
        for b in 3..=64u32 {
            let bb = u128::from(b);
            for l in 2..=16u32 {
                if bb.pow(l - 1) >= 1 << 85 {
                    break;
                }
                let (lo, hi) = (bb.pow(l - 1), (bb.pow(l) - 1).min((1 << 85) - 1));
                let mut ns = adversarial(b, l);
                for _ in 0..6 {
                    let x = (u128::from(xorshift(&mut r)) << 64) | u128::from(xorshift(&mut r));
                    ns.push(lo + x % (hi - lo + 1));
                }
                for n in ns {
                    for &(f0, k2) in SHAPES {
                        let Some((check, s2, s3)) = check_for(n, b, f0, k2) else {
                            continue;
                        };
                        let w = bb.pow(f0);
                        let (p, r0) = (n / w, u32::try_from(n % w).unwrap());
                        for _ in 0..4 {
                            // Short middles half the time, so that their
                            // digits are often distinct; any middle otherwise.
                            let cap = if xorshift(&mut r).is_multiple_of(2) {
                                8
                            } else {
                                48
                            };
                            let span = |s: u32, r: &mut u64| {
                                k2 + u32::try_from(xorshift(r) % u64::from((s - k2 + 1).min(cap)))
                                    .unwrap()
                            };
                            let ends = [span(s2, &mut r), span(s3, &mut r)];
                            let middle = check.count(p, r0, 0, ends, false);
                            let seen = if xorshift(&mut r).is_multiple_of(2) {
                                !middle.mask & xorshift(&mut r)
                            } else {
                                xorshift(&mut r)
                            };
                            let want = !check.count(p, r0, seen, ends, false).repeat;
                            assert_eq!(
                                check.nice(p, r0, seen, 0, ends),
                                want,
                                "b{b} n = {n} f0 {f0} k2 {k2} ends {ends:?} seen {seen:#x}"
                            );
                            runs += 1;
                            if want {
                                yes[check.nl as usize] += 1;
                            }
                        }
                    }
                }
            }
        }
        eprintln!("{runs} verdicts; true by limb count (0 = long division): {yes:?}");
        assert!(yes[1..].iter().all(|&y| y > 0), "{yes:?}");
    }

    /// What the check is given for `n` with its scan ending at `ends`: the
    /// digits below `k2` and those at and above `ends`, if each set is
    /// distinct and the two are disjoint (as the join ensures); `None`
    /// otherwise.
    fn seeds(d2: &[u32], d3: &[u32], k2: u32, ends: [u32; 2]) -> Option<(u64, u64)> {
        let k = k2 as usize;
        let low: Vec<u32> = d2[..k].iter().chain(&d3[..k]).copied().collect();
        let top: Vec<u32> = d2[ends[0] as usize..]
            .iter()
            .chain(&d3[ends[1] as usize..])
            .copied()
            .collect();
        let (lm, tm) = (bits(&low), bits(&top));
        (distinct(&low) && distinct(&top) && lm & tm == 0).then_some((lm, tm))
    }

    /// The seeded check decides what the client's full check decides
    /// (`get_is_nice`), and in count mode finds the client's count of
    /// distinct digits (`get_num_unique_digits`): on every `n` of the bands
    /// of bases 10-22 whose count is `b − 2` or more (the near-nice ones,
    /// 69 among them, which must be found), and on the band edges, `n²`'s
    /// and `n³`'s digit-length edges, adversarial and random `n` of bases
    /// 23-64; seeded with their known digits at random ends, the scan to
    /// the top (no top digits), and the scan ending at `k2`.
    #[test]
    fn the_seeded_check_decides_what_the_full_check_decides() {
        let mut r = 0x6A09_E667_F3BC_C908u64;
        let (mut decided, mut nice, mut near) = (0u64, 0u64, 0u64);
        let mut run = |n: u128, b: u32, r: &mut u64| {
            let (d2, d3) = digits(n, b);
            for &(f0, k2) in SHAPES {
                let Some((check, s2, s3)) = check_for(n, b, f0, k2) else {
                    continue;
                };
                let w = u128::from(b).pow(f0);
                let (p, r0) = (n / w, u32::try_from(n % w).unwrap());
                let mid = [
                    k2 + u32::try_from(xorshift(r) % u64::from(s2 - k2 + 1)).unwrap(),
                    k2 + u32::try_from(xorshift(r) % u64::from(s3 - k2 + 1)).unwrap(),
                ];
                for ends in [mid, [s2, s3], [k2, k2]] {
                    let Some((low, top)) = seeds(&d2, &d3, k2, ends) else {
                        continue;
                    };
                    let want = get_is_nice(n, b);
                    let got = check.nice(p, r0, low, top, ends);
                    assert_eq!(got, want, "b{b} n = {n} f0 {f0} k2 {k2} ends {ends:?}");
                    for long in [false, true] {
                        let c = check.count(p, r0, low | top, ends, long);
                        assert_eq!(
                            c.mask.count_ones(),
                            get_num_unique_digits(n, b),
                            "b{b} n = {n}"
                        );
                        assert_eq!(c.repeat, !want, "b{b} n = {n} ends {ends:?}");
                    }
                    decided += 1;
                    nice += u64::from(want);
                }
            }
        };
        for b in 10..=22u32 {
            let Some(band) = get_base_range_u128(b).unwrap() else {
                continue;
            };
            for n in band.start()..band.end() {
                if get_num_unique_digits(n, b) + 2 >= b {
                    run(n, b, &mut r);
                    near += 1;
                }
            }
        }
        for b in 23..=64u32 {
            let Some(band) = get_base_range_u128(b).unwrap() else {
                continue;
            };
            let (lo, hi) = (band.start(), band.end());
            let l = ndigits(lo, b);
            let mut ns = vec![lo, lo + 1, hi - 1, hi - 2];
            ns.extend(
                adversarial(b, l)
                    .into_iter()
                    .filter(|n| (lo..hi).contains(n)),
            );
            // Just below n² or n³ gaining a digit (outside the band too).
            for (e, root) in [
                (2 * l - 1, 2.0),
                (2 * l, 2.0),
                (3 * l - 2, 3.0),
                (3 * l - 1, 3.0),
            ] {
                // Near b^(e/root), within the f64 estimate's error.
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let x = (f64::from(b).powf(f64::from(e) / root)) as u128;
                for n in x.saturating_sub(3)..x + 4 {
                    if n > 0 && ndigits(n, b) == l && n < 1 << 85 {
                        ns.push(n);
                    }
                }
            }
            for _ in 0..3000 {
                let x = (u128::from(xorshift(&mut r)) << 64) | u128::from(xorshift(&mut r));
                ns.push(lo + x % (hi - lo));
            }
            for n in ns {
                run(n, b, &mut r);
            }
        }
        eprintln!("{decided} decided ({nice} nice), {near} near-nice n at bases 10-22");
        assert!(
            nice >= 1 && near > 50 && decided > 100_000,
            "{decided} {nice} {near}"
        );
    }

    /// Base 10's nice number in the shape of the join's base-10 window:
    /// seeded with its six known digits, found with or without its top's.
    #[test]
    fn sixty_nine_is_nice() {
        let check = Check::new(10, 2, 4, 6, 0, 3, 69).unwrap();
        assert!(check.columnar());
        // 69² = 4761, 69³ = 328509.
        let low = bits(&[1, 6, 7, 9, 0, 5]);
        assert!(check.nice(69, 0, low, 0, [4, 6]));
        assert!(check.nice(69, 0, low, bits(&[4, 3, 2]), [3, 4]));
        assert!(!check.nice(68, 0, low, 0, [4, 6]));
    }

    /// Every production shape (bases 40-64, `f0 = 3`, `k2 = 8`, the band's
    /// digit counts) takes the column form for every `P` of the band.
    #[test]
    fn production_shapes_take_the_columns() {
        for b in 40..=64u32 {
            let Some(band) = get_base_range_u128(b).unwrap() else {
                continue;
            };
            let (d2, d3) = digits(band.start(), b);
            let l = ndigits(band.start(), b);
            let (s2, s3) = (
                u32::try_from(d2.len()).unwrap(),
                u32::try_from(d3.len()).unwrap(),
            );
            let check = Check::new(b, l, s2, s3, 3, 8, band.end() / u128::from(b).pow(3)).unwrap();
            assert!(check.columnar(), "b{b} L{l}");
        }
    }
}
