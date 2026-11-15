//! The overlap join's full check (`cubecl_join::check_kernel`): its shape
//! and its constants, and (in the tests) a host mirror of its arithmetic.
//!
//! A survivor of the join is `n = P·b^f0 + r0`, with `P` from its top's
//! record, whose word 6 is `w6 = P mod b^(k2 − f0)`. The check writes `n` in
//! base-b² limbs (two digits each) straight from `r0`, `w6` and
//! `H = (P − w6) / b^(k2 − f0) = ⌊n / b^k2⌋`, a division that is exact, so a
//! shift and a multiply by an inverse. It squares and cubes `n` by schoolbook
//! columns of u32 products, every column below 2^31 ([`CheckShape::new`]),
//! and takes the limbs and their digits out of the carry normalisation with
//! multiply-high quotients ([`magic`]): no long division of a binary number,
//! and no `/` or `%` anywhere. A survivor goes on to `n³`'s upper limbs only
//! if the digits of `n²` and the low digits of `n³` are distinct (the gate,
//! [`CheckShape::gl`]): a necessary condition that few survivors pass, so
//! most planes of the kernel skip the rest of `n³`.

use anyhow::{Result, ensure};

/// `n³`'s low digits the gate takes with `n²`'s (rounded down to whole
/// limbs below `n³`'s top limb, [`CheckShape::gl`]).
pub const GATE_DIGITS: u32 = 10;

/// The multiplier and shift `(m, s)` with `⌊x / d⌋ = mul_hi(x, m) >> s` for
/// every `x < 2^31`, for `d >= 3` not a power of two: `s = ⌊log2 d⌋` and
/// `m = ⌈2^(32+s) / d⌉ < 2^32` (as `2^s < d`). With `e = m·d − 2^(32+s)`,
/// `0 < e < d < 2^(s+1)`, and `x·m / 2^(32+s) = x/d + x·e / (d·2^(32+s))`,
/// whose second term is below `1/d` when `x·e < 2^(32+s)`, which holds for
/// `x < 2^31`: the quotient is exact (the tests' `magic_bound` is where that
/// sufficient condition ends, at or above 2^31 for every divisor).
///
/// # Panics
/// If `d < 3` or `d` is a power of two.
#[must_use]
#[allow(clippy::cast_lossless)] // `u64::from` is not const
pub const fn magic(d: u32) -> (u32, u32) {
    assert!(
        d >= 3 && !d.is_power_of_two(),
        "magic: d >= 3, not a power of 2"
    );
    let s = d.ilog2();
    let k = 32 + s;
    let m = (1u64 << k).div_ceil(d as u64);
    assert!(m < 1 << 32, "magic: m fits u32");
    #[allow(clippy::cast_possible_truncation)]
    (m as u32, s)
}

/// The first `x` the sufficient condition `x·e < 2^(32+s)` of [`magic`]
/// fails at for divisor `d` (at least 2^31).
#[cfg(test)]
#[must_use]
#[allow(clippy::cast_lossless)] // `u64::from` is not const
pub(crate) const fn magic_bound(d: u32) -> u64 {
    let (m, s) = magic(d);
    let e = m as u64 * d as u64 - (1u64 << (32 + s));
    (1u64 << (32 + s)).div_ceil(e)
}

/// `⌊x / d⌋` as the kernel computes it: [`magic`], or a shift for a power
/// of two.
#[cfg(test)]
#[must_use]
pub(crate) fn qdiv(x: u32, d: u32) -> u32 {
    if d.is_power_of_two() {
        return x >> d.trailing_zeros();
    }
    debug_assert!(x < 1 << 31, "qdiv({x}, {d}): outside the magic's range");
    let (m, s) = magic(d);
    #[allow(clippy::cast_possible_truncation)]
    let hi = ((u64::from(x) * u64::from(m)) >> 32) as u32;
    hi >> s
}

/// `o^-1 mod 2^64` for odd `o` (Newton's iteration, 3 → 96 correct bits).
///
/// # Panics
/// If `o` is even.
#[must_use]
pub const fn inverse_mod_2_64(o: u64) -> u64 {
    assert!(o % 2 == 1, "inverse of an even number");
    let mut x = o;
    let mut i = 0;
    while i < 5 {
        x = x.wrapping_mul(2u64.wrapping_sub(o.wrapping_mul(x)));
        i += 1;
    }
    x
}

/// The comptime constants of one field's check: base `b`, digits `l` of
/// `n`, `s2` of `n²` and `s3` of `n³` (constant over the field), the
/// join's `f0` (`n = P·b^f0 + r0`) and `k2` (the record's word 6 is
/// `P mod b^(k2 − f0)`), and the gate's digit count `g`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CheckShape {
    pub b: u32,
    pub l: u32,
    pub s2: u32,
    pub s3: u32,
    pub f0: u32,
    pub k2: u32,
    pub g: u32,
}

impl CheckShape {
    /// # Errors
    /// A shape the kernel's bounds do not cover: a base outside 3-64,
    /// digit counts that no `n` of `l` digits has, a record residue or
    /// `r0` of 2^31 or more, or columns that could reach 2^31.
    // b, l, s2, s3, f0, k2, g are the fields' names.
    #[allow(clippy::many_single_char_names)]
    pub fn new(b: u32, l: u32, s2: u32, s3: u32, f0: u32, k2: u32, g: u32) -> Result<Self> {
        ensure!((3..=64).contains(&b), "check: base {b} outside 3-64");
        ensure!((1..=48).contains(&l), "check: {l} digits");
        ensure!(
            (2 * l - 1..=2 * l).contains(&s2) && (3 * l - 2..=3 * l).contains(&s3),
            "check: n of {l} digits has no square of {s2} or cube of {s3} digits"
        );
        ensure!(f0 < k2, "check: f0 {f0} >= k2 {k2}");
        ensure!(g >= 2 && g.is_multiple_of(2), "check: gate of {g} digits");
        let bb = u128::from(b);
        ensure!(
            bb.pow(f0) < 1 << 31 && bb.pow(k2 - f0) < 1 << 31,
            "check: b^f0 or b^(k2 - f0) reaches 2^31 at base {b}"
        );
        let sh = Self {
            b,
            l,
            s2,
            s3,
            f0,
            k2,
            g,
        };
        // A column of n² or n³ has at most nl products of limbs below b²,
        // and a carry below nl·b² (by induction), so it is below
        // nl·(b⁴ + b²).
        let col = u128::from(sh.nl()) * bb.pow(4) + u128::from(sh.nl()) * bb * bb;
        ensure!(
            col < 1 << 31,
            "check: columns reach 2^31 at base {b}, {l} digits"
        );
        Ok(sh)
    }

    /// Limbs (base b²) of `n`, `n²` and `n³`.
    #[must_use]
    pub fn nl(&self) -> u32 {
        self.l.div_ceil(2)
    }
    #[must_use]
    pub fn ql(&self) -> u32 {
        self.s2.div_ceil(2)
    }
    #[must_use]
    pub fn cl(&self) -> u32 {
        self.s3.div_ceil(2)
    }
    /// `n³`'s limbs the gate takes: `g / 2`, below `n³`'s top limb.
    #[must_use]
    pub fn gl(&self) -> u32 {
        (self.g / 2).min(self.cl() - 1)
    }

    /// Part `p` of `n`'s digits (0: `r0`, 1: `w6`, 2: `H`) as (first digit,
    /// digits), clipped to `l`.
    #[must_use]
    pub fn part(&self, p: u32) -> (u32, u32) {
        let (f0, k2) = (self.f0.min(self.l), self.k2.min(self.l));
        match p {
            0 => (0, f0),
            1 => (f0, k2 - f0),
            _ => (k2, self.l - k2),
        }
    }

    /// Whether `H` can reach 2^31 (only in the tests' small windows; every
    /// production field has `H < b^5`).
    #[must_use]
    pub fn h_wide(&self) -> bool {
        u128::from(self.b).pow(self.part(2).1) > 1 << 31
    }

    /// `b^(k2 − f0) = 2^z·o` with `o` odd: `(z, o^-1 mod 2^64)`.
    #[must_use]
    pub fn h_div(&self) -> (u32, u64) {
        let d = u64::from(self.b).pow(self.k2 - self.f0);
        let z = d.trailing_zeros();
        (z, inverse_mod_2_64(d >> z))
    }
}

/// What the kernel computes for one survivor (the mirror's output).
#[cfg(test)]
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Mirror {
    /// `n`, `n²` and `n³` in base-b² limbs, least significant first.
    pub(crate) nn: Vec<u32>,
    pub(crate) sq: Vec<u32>,
    pub(crate) cu: Vec<u32>,
    /// Digits of `n²`, those the gate sees (`n²` and `n³`'s low `2·gl`
    /// digits), and all of `n²` and `n³`'s, as bits.
    pub(crate) sq_mask: u64,
    pub(crate) gate_mask: u64,
    pub(crate) mask: u64,
}

#[cfg(test)]
impl Mirror {
    /// Whether the survivor passes the gate.
    #[must_use]
    pub(crate) fn gate(&self, sh: &CheckShape) -> bool {
        self.gate_mask.count_ones() == sh.s2 + 2 * sh.gl()
    }
}

#[cfg(test)]
fn add(x: u32, y: u32) -> u32 {
    x.checked_add(y).expect("a column below 2^31")
}

#[cfg(test)]
fn mul(x: u32, y: u32) -> u32 {
    x.checked_mul(y).expect("a product below 2^31")
}

/// The kernel's arithmetic for the survivor `(P, w6, r0)` on the host, step
/// for step: the same limbs, products, columns, quotients and masks. Every
/// sum and product is checked for u32 overflow.
///
/// # Panics
/// On an overflow, or a part that does not fit its digits: a shape or a
/// survivor outside the kernel's bounds.
#[cfg(test)]
#[must_use]
// The kernel's names: b, bb, nl, ql, cl, gl, x, q, v; its steps in order.
#[allow(
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::too_many_lines
)]
pub(crate) fn mirror(sh: &CheckShape, p: u64, w6: u32, r0: u32) -> Mirror {
    let (b, bb) = (sh.b, sh.b * sh.b);
    let (nl, ql, cl, gl) = (
        sh.nl() as usize,
        sh.ql() as usize,
        sh.cl() as usize,
        sh.gl() as usize,
    );
    // H = (P − w6) / b^(k2 − f0), exact: a shift and an inverse.
    let (z, inv) = sh.h_div();
    let x = (p - u64::from(w6)) >> z;
    let h = x.wrapping_mul(inv);
    // n's limbs, part by part; a part that starts at an odd digit completes
    // the limb whose low digit the part before left pending.
    let mut nn = vec![0u32; nl];
    let mut pend = 0u32;
    for part in 0..3 {
        let (o, c) = sh.part(part);
        if c == 0 {
            continue;
        }
        // The parts' values: r0 < b^f0, w6 < b^(k2 − f0) (both < 2^31), H.
        let mut v: u64 = match part {
            0 => u64::from(r0),
            1 => u64::from(w6),
            _ => h,
        };
        // A wide H (2^31 or more; test windows only) is taken apart by
        // long division over 16-bit chunks, as the kernel does.
        let wide = part == 2 && sh.h_wide();
        let div = |v: u64, d: u32| -> (u64, u32) {
            if wide {
                wide_div(v, d)
            } else {
                let v32 = u32::try_from(v).expect("a narrow part");
                let q = qdiv(v32, d);
                (u64::from(q), v32 - q * d)
            }
        };
        let (mut o2, mut c2) = (o, c);
        if o % 2 == 1 {
            let hi = if c == 1 {
                u32::try_from(v).expect("a digit")
            } else {
                let (q, r) = div(v, b);
                v = q;
                r
            };
            nn[(o / 2) as usize] = pend + hi * b;
            o2 += 1;
            c2 -= 1;
        }
        for q in 0..c2 / 2 {
            let j = (o2 / 2 + q) as usize;
            if q + 1 == c2 / 2 && c2.is_multiple_of(2) {
                nn[j] = u32::try_from(v).expect("the part's last limb");
            } else {
                let (qq, r) = div(v, bb);
                nn[j] = r;
                v = qq;
            }
        }
        if c2 % 2 == 1 {
            let d = u32::try_from(v).expect("the part's last digit");
            if o2 + c2 == sh.l {
                nn[((sh.l - 1) / 2) as usize] = d;
            } else {
                pend = d;
            }
        }
    }
    // n²: column k is twice the products i < j, i + j = k, and the square of
    // limb k/2, plus the carry; one quotient by b² per column.
    let mut sq = vec![0u32; ql];
    let mut carry = 0u32;
    for (k, out) in sq.iter_mut().enumerate() {
        let mut s = 0u32;
        for i in 0..nl {
            let j = k.wrapping_sub(i);
            if i < j && j < nl {
                s = add(s, mul(nn[i], nn[j]));
            }
        }
        let mut x = add(carry, mul(s, 2));
        if k.is_multiple_of(2) && k / 2 < nl {
            x = add(x, mul(nn[k / 2], nn[k / 2]));
        }
        if k + 1 == ql {
            *out = x;
        } else {
            let q = qdiv(x, bb);
            *out = x - q * bb;
            carry = q;
        }
    }
    // n³ = n²·n: the gate's columns first, then the rest from its carry.
    let mut cu = vec![0u32; cl];
    let mut carry = 0u32;
    for (k, out) in cu.iter_mut().enumerate() {
        let mut x = carry;
        for (j, &a) in nn.iter().enumerate() {
            let i = k.wrapping_sub(j);
            if i < ql {
                x = add(x, mul(sq[i], a));
            }
        }
        if k + 1 == cl {
            *out = x;
        } else {
            let q = qdiv(x, bb);
            *out = x - q * bb;
            carry = q;
        }
    }
    // Digits: two per limb (one in a top limb of an odd digit count).
    let put = |mask: &mut u64, y: u32, two: bool| {
        if two {
            let hi = qdiv(y, b);
            *mask |= (1u64 << (y - hi * b)) | (1u64 << hi);
        } else {
            *mask |= 1u64 << y;
        }
    };
    let mut mask = 0u64;
    for (k, &y) in sq.iter().enumerate() {
        put(&mut mask, y, k + 1 < ql || sh.s2.is_multiple_of(2));
    }
    let sq_mask = mask;
    for &y in &cu[..gl] {
        put(&mut mask, y, true);
    }
    let gate_mask = mask;
    for (k, &y) in cu.iter().enumerate().skip(gl) {
        put(&mut mask, y, k + 1 < cl || sh.s3.is_multiple_of(2));
    }
    Mirror {
        nn,
        sq,
        cu,
        sq_mask,
        gate_mask,
        mask,
    }
}

/// `(⌊v / d⌋, v mod d)` for a u64 `v` and `d <= 4096`, as the kernel takes a
/// wide `H` apart: long division over 16-bit chunks, each dividend below
/// `d·2^16 <= 2^28`, by `qdiv`.
#[cfg(test)]
#[must_use]
#[allow(clippy::many_single_char_names)] // the kernel's names
pub(crate) fn wide_div(v: u64, d: u32) -> (u64, u32) {
    let mut rem = 0u32;
    let mut q = 0u64;
    for i in (0..4).rev() {
        #[allow(clippy::cast_possible_truncation)]
        let c = ((v >> (16 * i)) & 0xFFFF) as u32;
        let x = (rem << 16) | c;
        let qc = qdiv(x, d);
        rem = x - qc * d;
        q |= u64::from(qc) << (16 * i);
    }
    (q, rem)
}

#[cfg(test)]
#[allow(clippy::many_single_char_names, clippy::similar_names)]
pub(crate) mod tests {
    use super::*;
    use malachite::base::num::arithmetic::traits::Pow;
    use malachite::base::num::conversion::traits::Digits;
    use malachite::natural::Natural;

    fn digits(x: &Natural, b: u32) -> Vec<u32> {
        x.to_digits_asc(&b)
    }

    /// Base-b² limbs of `x` from its base-b digits, `n` of them.
    fn limbs(d: &[u32], b: u32, n: usize) -> Vec<u32> {
        (0..n)
            .map(|j| {
                let lo = d.get(2 * j).copied().unwrap_or(0);
                let hi = d.get(2 * j + 1).copied().unwrap_or(0);
                lo + hi * b
            })
            .collect()
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

    /// Every divisor the kernel divides by: b and b² at every base 3-64.
    fn divisors() -> Vec<u32> {
        (3..=64u32)
            .flat_map(|b| [b, b * b])
            .filter(|d| !d.is_power_of_two())
            .collect()
    }

    /// The bound of [`magic`] holds for every divisor, and the quotient is
    /// exact at the worst element of every block of `d` below the inputs the
    /// kernel gives it: `mul_hi(x, m) >> s` is monotone in `x` and never below
    /// `⌊x/d⌋` (as `m·d >= 2^(32+s)`), so if it is exact at `q·d + d − 1` it
    /// is exact on all of `[q·d, q·d + d)`. Inputs: by b², every column and
    /// wide-H chunk (below 2^28); by b, every limb's split (below b²) and the
    /// production record parts (`w6 < b^5`, `r0 < b^3`) at bases 40-64.
    #[test]
    fn magic_quotients_are_exact_on_the_kernels_inputs() {
        for d in divisors() {
            assert!(magic_bound(d) >= 1 << 31, "d = {d}: {}", magic_bound(d));
        }
        let check = |d: u32, below: u64| {
            let (m, s) = magic(d);
            let d64 = u64::from(d);
            let mut q = 0u64;
            while q * d64 < below {
                let x = (q * d64 + d64 - 1).min(below - 1);
                let got = ((x * u64::from(m)) >> 32) >> s;
                assert_eq!(got, x / d64, "d = {d}, x = {x}");
                q += 1;
            }
        };
        for b in 3..=64u32 {
            let bb = b * b;
            if !bb.is_power_of_two() {
                check(bb, 1 << 28);
            }
            if !b.is_power_of_two() {
                check(b, u64::from(bb));
                if b >= 40 {
                    check(b, u64::from(b).pow(5));
                }
            }
        }
    }

    /// Opt-in (`NICE_TEST_CHECK_EXHAUSTIVE=1`; minutes on all cores): every
    /// `x < 2^31` for every divisor of [`divisors`].
    #[test]
    #[ignore = "exhaustive, minutes on all cores; opt-in, NICE_TEST_CHECK_EXHAUSTIVE=1"]
    fn magic_quotients_are_exact_everywhere() {
        if std::env::var("NICE_TEST_CHECK_EXHAUSTIVE").is_err() {
            eprintln!("skipping: set NICE_TEST_CHECK_EXHAUSTIVE=1 to run every x below 2^31");
            return;
        }
        let ds = divisors();
        let threads = std::thread::available_parallelism().map_or(4, std::num::NonZero::get);
        let next = std::sync::atomic::AtomicUsize::new(0);
        std::thread::scope(|sc| {
            for _ in 0..threads {
                sc.spawn(|| {
                    loop {
                        let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let Some(&d) = ds.get(i) else { break };
                        let (m, s) = magic(d);
                        for x in 0..1u32 << 31 {
                            #[allow(clippy::cast_possible_truncation)]
                            let got = (((u64::from(x) * u64::from(m)) >> 32) as u32) >> s;
                            assert_eq!(got, x / d, "d = {d}, x = {x}");
                        }
                    }
                });
            }
        });
        println!("{} divisors exact on every x below 2^31", ds.len());
    }

    #[test]
    fn inverses_and_wide_division_are_exact() {
        let mut r = 0x9E37_79B9_7F4A_7C15u64;
        for b in 3..=64u64 {
            for e in 1..=10u32 {
                let d = b.pow(e);
                let z = d.trailing_zeros();
                let inv = inverse_mod_2_64(d >> z);
                assert_eq!((d >> z).wrapping_mul(inv), 1);
                for _ in 0..200 {
                    let h = xorshift(&mut r) >> (xorshift(&mut r) % 64);
                    let Some(p) = h.checked_mul(d) else { continue };
                    assert_eq!((p >> z).wrapping_mul(inv), h, "b^{e} = {d}");
                }
            }
            let bb = u32::try_from(b * b).unwrap();
            for d in [u32::try_from(b).unwrap(), bb] {
                for _ in 0..2000 {
                    let v = xorshift(&mut r) >> (xorshift(&mut r) % 64);
                    let rem = u32::try_from(v % u64::from(d)).unwrap();
                    assert_eq!(wide_div(v, d), (v / u64::from(d), rem));
                }
                assert_eq!(wide_div(u64::MAX, d).0, u64::MAX / u64::from(d));
            }
        }
    }

    /// The shapes the tests take `n` in: the join's production shape
    /// (f0 = 3, k2 = 8) and the test windows' (f0 0-3, k2 up to 8).
    fn shapes_for(n: u128, b: u32, all: bool) -> Vec<CheckShape> {
        let l = ndigits(n, b);
        let nat = Natural::from(n);
        let s2 = u32::try_from(digits(&(&nat * &nat), b).len()).unwrap();
        let s3 = u32::try_from(digits(&nat.pow(3), b).len()).unwrap();
        let mut v = Vec::new();
        let pairs: &[(u32, u32)] = if all {
            &[
                (3, 8),
                (0, 3),
                (1, 4),
                (1, 6),
                (2, 5),
                (2, 6),
                (2, 7),
                (3, 6),
                (3, 7),
            ]
        } else {
            &[(3, 8), (0, 3), (2, 6), (3, 7)]
        };
        for &(f0, k2) in pairs {
            // The records hold P = n / b^f0 in two words.
            if n / u128::from(b).pow(f0) >= 1 << 64 {
                continue;
            }
            if let Ok(sh) = CheckShape::new(b, l, s2, s3, f0, k2, GATE_DIGITS) {
                v.push(sh);
            }
        }
        v
    }

    /// What the check decides about `n` in shape `sh`, from `Natural`
    /// digits: the distinct digits of n² and n³ (as
    /// `get_num_unique_digits`), whether the gate passes (n²'s digits and
    /// n³'s low `2·gl` distinct), the digit mask, and the gate's digits as
    /// a mask (n²'s and n³'s low `2·gl`).
    #[cfg_attr(not(feature = "cubecl"), allow(dead_code))] // the device tests' reference
    pub(crate) fn reference(sh: &CheckShape, n: u128) -> (u32, bool, u64, u64) {
        let b = sh.b;
        let nat = Natural::from(n);
        let (d2, d3) = (digits(&(&nat * &nat), b), digits(&(&nat * &nat * &nat), b));
        let mask = d2.iter().chain(&d3).fold(0u64, |a, &x| a | 1u64 << x);
        let uniq = mask.count_ones();
        // The client's count (its fast paths hold n³ in 128 or 256 bits,
        // enough for the base's own range, where s2 + s3 = b).
        if sh.s2 + sh.s3 == b {
            assert_eq!(
                uniq,
                crate::client_process::get_num_unique_digits(n, b),
                "b{b} n = {n}"
            );
        }
        let mut seen = d2.clone();
        seen.extend_from_slice(&d3[..2 * sh.gl() as usize]);
        let gate_mask = seen.iter().fold(0u64, |a, &x| a | 1u64 << x);
        seen.sort_unstable();
        (uniq, seen.windows(2).all(|w| w[0] != w[1]), mask, gate_mask)
    }

    /// The mirror against `Natural` digits for `n` in shape `sh`: every limb
    /// of n, n², n³, the masks, the count of distinct digits (as
    /// `get_num_unique_digits`), and the gate's predicate by its definition.
    fn assert_mirror(sh: &CheckShape, n: u128) {
        let b = sh.b;
        let w = u128::from(b).pow(sh.f0);
        let p = u64::try_from(n / w).expect("P < 2^64");
        let r0 = u32::try_from(n % w).unwrap();
        let w6 = u32::try_from(u128::from(p) % u128::from(b).pow(sh.k2 - sh.f0)).unwrap();
        let m = mirror(sh, p, w6, r0);
        let nat = Natural::from(n);
        let (d1, d2, d3) = (
            digits(&nat, b),
            digits(&(&nat * &nat), b),
            digits(&(&nat * &nat * &nat), b),
        );
        assert_eq!((d2.len(), d3.len()), (sh.s2 as usize, sh.s3 as usize));
        assert_eq!(
            m.nn,
            limbs(&d1, b, sh.nl() as usize),
            "b{b} n = {n} {sh:?}: n's limbs"
        );
        assert_eq!(
            m.sq,
            limbs(&d2, b, sh.ql() as usize),
            "b{b} n = {n} {sh:?}: n²"
        );
        assert_eq!(
            m.cu,
            limbs(&d3, b, sh.cl() as usize),
            "b{b} n = {n} {sh:?}: n³"
        );
        let bits = |d: &[u32]| d.iter().fold(0u64, |a, &x| a | 1u64 << x);
        let gd = 2 * sh.gl() as usize;
        assert_eq!(m.mask, bits(&d2) | bits(&d3), "b{b} n = {n}: mask");
        assert_eq!(
            m.gate_mask,
            bits(&d2) | bits(&d3[..gd]),
            "b{b} n = {n}: gate mask"
        );
        if sh.s2 + sh.s3 == b {
            assert_eq!(
                m.mask.count_ones(),
                crate::client_process::get_num_unique_digits(n, b),
                "b{b} n = {n}: distinct digits"
            );
        }
        let mut seen = d2.clone();
        seen.extend_from_slice(&d3[..gd]);
        seen.sort_unstable();
        let distinct = seen.windows(2).all(|w| w[0] != w[1]);
        assert_eq!(m.gate(sh), distinct, "b{b} n = {n}: gate");
        // The gate is necessary: a nice n passes it.
        if m.mask.count_ones() == b && sh.s2 + sh.s3 == b {
            assert!(m.gate(sh), "b{b} n = {n}: a nice n fails the gate");
        }
    }

    /// Adversarial `n` of `l` digits: every digit b − 1 (the largest columns),
    /// the smallest, 0 and b − 1 alternating either way, b − 1 but one digit,
    /// and the parts' extremes (r0, w6, H at 0 and at their largest).
    pub(crate) fn adversarial(b: u32, l: u32) -> Vec<u128> {
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
        let mut v = vec![top, low, low + 1, alt(0), alt(1)];
        for i in 0..l {
            v.push(top - u128::from(b - 1) * bb.pow(i));
            v.push(low + bb.pow(i) - 1);
        }
        for f in [3u32, 8] {
            if f < l {
                let w = bb.pow(f);
                v.push(top / w * w);
                v.push(low / w * w + w - 1);
            }
        }
        v.retain(|&n| (low..=top).contains(&n) && n < 1 << 96);
        v
    }

    #[test]
    fn mirror_matches_natural_at_every_base() {
        let mut r = 0x2545_F491_4F6C_DD1Du64;
        let mut runs = 0u64;
        for b in 10..=64u32 {
            let bb = u128::from(b);
            for l in 2..=24u32 {
                if bb.pow(l - 1) >= 1 << 96 {
                    break;
                }
                let (lo, hi) = (bb.pow(l - 1), (bb.pow(l) - 1).min((1 << 96) - 1));
                let mut ns = adversarial(b, l);
                for _ in 0..16 {
                    let x = (u128::from(xorshift(&mut r)) << 64) | u128::from(xorshift(&mut r));
                    ns.push(lo + x % (hi - lo + 1));
                }
                for n in ns {
                    for sh in shapes_for(n, b, false) {
                        assert_mirror(&sh, n);
                        runs += 1;
                    }
                }
            }
        }
        assert!(runs > 20_000, "{runs} shapes tried");
    }

    /// The gate against its definition (n²'s digits and n³'s low `2·gl`
    /// distinct) on `n` whose square has distinct digits, where it decides
    /// something: small bases, where those are common, every shape. And the
    /// known nice number passes it (no other nice number is in range).
    #[test]
    fn gate_matches_its_definition() {
        let sh = CheckShape::new(10, 2, 4, 6, 0, 3, GATE_DIGITS).unwrap();
        let m = mirror(&sh, 69, 69, 0);
        assert!(m.gate(&sh) && m.mask.count_ones() == 10, "69 in base 10");
        let (mut passed, mut decided) = (0u32, 0u32);
        for b in 10..=24u32 {
            let bb = u128::from(b);
            for l in 2..=7u32 {
                let lo = bb.pow(l - 1);
                let step = (bb.pow(l) - lo) / 1000 + 1;
                for n in (lo..bb.pow(l)).step_by(usize::try_from(step).unwrap()) {
                    for sh in shapes_for(n, b, true) {
                        let w = bb.pow(sh.f0);
                        let p = u64::try_from(n / w).unwrap();
                        let w6 = u32::try_from(u128::from(p) % bb.pow(sh.k2 - sh.f0)).unwrap();
                        let m = mirror(&sh, p, w6, u32::try_from(n % w).unwrap());
                        if m.sq_mask.count_ones() == sh.s2 {
                            decided += 1;
                            assert_mirror(&sh, n);
                            passed += u32::from(m.gate(&sh));
                        }
                    }
                }
            }
        }
        assert!(
            decided > 1000 && passed > 100,
            "{decided} decided, {passed} passed"
        );
    }

    #[test]
    fn shapes_of_the_join_fields_are_supported() {
        // Every production band 40-64 (t = L − 3, k = 6, p = 2: f0 = 3, k2 = 8)
        // at its digit lengths, with H below 2^31.
        let mut shapes = 0;
        for b in 40..=64u32 {
            for l in 8..=13u32 {
                for (s2, s3) in
                    (2 * l - 1..=2 * l).flat_map(|s2| (3 * l - 2..=3 * l).map(move |s3| (s2, s3)))
                {
                    if s2 + s3 != b {
                        continue;
                    }
                    shapes += 1;
                    let sh = CheckShape::new(b, l, s2, s3, 3, 8, GATE_DIGITS).expect("a shape");
                    assert!(!sh.h_wide(), "b{b} L{l}");
                    assert_eq!(sh.gl(), 5);
                }
            }
        }
        assert!(shapes >= 25, "{shapes}");
    }
}
