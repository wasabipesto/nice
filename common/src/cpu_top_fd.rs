//! The CPU join's top certificates by finite differences.
//!
//! [`crate::cpu_join`] certifies one key's top prefixes at a time, in
//! ascending order, and two prefixes of one key differ by a multiple of
//! `h = b^o` (`o = k − f0`: the partition's digits and the key digit are
//! `P`'s lowest `o`). The block of `P` is `[a, a + w − 1]` with `a = P·w`
//! and `w = b^f0`, so `a^j = P^j·b^(j·f0)`: the digits of `a²` and `a³` are
//! those of `P²` and `P³`, shifted up. Over `P, P + h, P + 2h, …` those
//! follow from additions, as in the detailed mode ([`crate::detailed_fd`])
//! with a step of `h` instead of 1:
//!
//! ```text
//! SQ = P²   SQ += DS   DS  = 2Ph + h²           DS  += 2h²
//! CU = P³   CU += DC   DC  = 3P²h + 3Ph² + h³   DC  += DDC
//!                      DDC = 6Ph² + 6h³         DDC += 6h³
//! ```
//!
//! The five are base-b digit arrays (one digit a byte, eight a word, added
//! with the detailed mode's biased add), and `2h²`, `6h³` are single digits
//! at positions `2o` and `3o`. The certificate also needs `d_j = e^j − a^j`
//! (`e = a + δ`, `δ = w − 1`). It steps in binary: with `H = h·w`,
//! `d2 += 2δH`, `d3 += Δ3` and `Δ3 = 3δH(2a + H + δ) += 6δH²`.
//!
//! [`TopFdSetup::cert`] is `Base::cert` with the digits read from the arrays
//! instead of divided out of `a^j`, `ndig(d_j)` from its bit length, and the
//! carry test screened in floating point. A prefix more than [`MAX_STEPS`]
//! steps above the last one (a key's first, or after a gap) seeds the
//! state: `P`, `P²` and `P³` are converted to digits with reciprocal
//! multiplies, and the differences are built from those by digit additions.

// The digit-array helpers are forced inline: they sit in the per-prefix loop.
#![allow(clippy::inline_always)]

use crate::cpu_join::SmallDiv;
use crate::overlap_join::{Base, Cert, Div64, W4, close_certificate, digit_range_mask};

/// The most steps the state walks to the next prefix; one farther away
/// seeds it instead.
const MAX_STEPS: u32 = 4;

/// Digits of every digit array: `P³` for every field the join takes, with
/// the word a carry out of the longest addend runs into.
const DIGITS: u32 = 48;
const NW: usize = (DIGITS / 8) as usize;

type Digits = [u64; NW];

/// One in every byte of a word.
const LO: u64 = 0x0101_0101_0101_0101;

/// [`crate::detailed_fd`]'s biased add of eight digits, with the bias
/// `256 − b` a runtime value: `a + r + cin` and the carry out of the top
/// byte. Every byte of `a` and `r` must be below `b`.
#[inline(always)]
fn add_word(a: u64, r: u64, cin: bool, bias: u64) -> (u64, bool) {
    let rb = r + bias * LO;
    let (t, c1) = a.overflowing_add(rb);
    let (t, c2) = t.overflowing_add(u64::from(cin));
    let cout = c1 | c2;
    let carried = (((t ^ a ^ rb) >> 8) & LO) | (u64::from(cout) << 56);
    (t - (carried ^ LO) * bias, cout)
}

/// `x += y` for `y` zero outside words `lo..hi`: those words, then the
/// carry into word `hi`, and on (rarely, past eight digits of `b − 1`).
/// The sum must fit `x`.
#[inline(always)]
fn add_digits(x: &mut Digits, y: &Digits, lo: usize, hi: usize, bias: u64) {
    let mut c = false;
    for i in lo..hi {
        let (t, co) = add_word(x[i], y[i], c, bias);
        x[i] = t;
        c = co;
    }
    let (t, co) = add_word(x[hi], 0, c, bias);
    x[hi] = t;
    if co {
        ripple(x, hi + 1, bias);
    }
}

/// Carry one into word `i` and up.
#[cold]
#[inline(never)]
fn ripple(x: &mut Digits, mut i: usize, bias: u64) {
    loop {
        let (t, c) = add_word(x[i], 0, true, bias);
        x[i] = t;
        if !c {
            return;
        }
        i += 1;
    }
}

/// `x + y` over every word (the seed's additions); the sum must fit.
fn sum(x: &Digits, y: &Digits, bias: u64) -> Digits {
    let mut out = [0u64; NW];
    let mut c = false;
    for (o, (&a, &r)) in out.iter_mut().zip(x.iter().zip(y)) {
        (*o, c) = add_word(a, r, c, bias);
    }
    debug_assert!(!c, "digit array overflow");
    out
}

/// `x·b^n`; the product must fit.
fn shl_digits(x: &Digits, n: u32) -> Digits {
    let (q, r) = ((n / 8) as usize, 8 * (n % 8));
    let mut out = [0u64; NW];
    for i in q..NW {
        out[i] = x[i - q] << r;
        if r > 0 && i > q {
            out[i] |= x[i - q - 1] >> (64 - r);
        }
    }
    out
}

/// The single digit `g` at position `n`.
fn digit_at(g: u64, n: u32) -> Digits {
    let mut out = [0u64; NW];
    out[(n / 8) as usize] = g << (8 * (n % 8));
    out
}

/// Digit `i` of `x`.
// A byte of the array: a digit below b <= 64.
#[allow(clippy::cast_possible_truncation)]
#[inline(always)]
fn digit(x: &Digits, i: u32) -> u8 {
    (x[(i / 8) as usize] >> (8 * (i % 8))) as u8
}

/// The value of a word of eight digits (below `b^8 <= 2^48`), by three
/// multiplies: digit pairs in 16-bit lanes, then quads in 32-bit lanes.
#[inline(always)]
fn word_value(w: u64, b: u64, b2: u64, b4: u64) -> u64 {
    const M8: u64 = 0x00FF_00FF_00FF_00FF;
    const M16: u64 = 0x0000_FFFF_0000_FFFF;
    let x = (w & M8) + ((w >> 8) & M8) * b;
    let y = (x & M16) + ((x >> 16) & M16) * b2;
    (y & 0xFFFF_FFFF) + (y >> 32) * b4
}

fn w4(x: u128) -> W4 {
    // The two halves of x.
    #[allow(clippy::cast_possible_truncation)]
    W4([x as u64, (x >> 64) as u64, 0, 0])
}

/// The bit length of `x`.
fn bit_len(x: &W4) -> usize {
    let mut i = 4;
    while i > 0 && x.0[i - 1] == 0 {
        i -= 1;
    }
    if i == 0 {
        0
    } else {
        64 * i - x.0[i - 1].leading_zeros() as usize
    }
}

/// `x` as f64, from its top two nonzero words: relative error below
/// `2^-51`.
#[allow(clippy::cast_precision_loss)]
#[inline(always)]
fn w4_f64(x: &W4) -> f64 {
    const TWO64: f64 = 18_446_744_073_709_551_616.0;
    let mut i = 3;
    while i > 0 && x.0[i] == 0 {
        i -= 1;
    }
    if i == 0 {
        return x.0[0] as f64;
    }
    let scale = if i == 3 {
        TWO64 * TWO64
    } else if i == 2 {
        TWO64
    } else {
        1.0
    };
    (x.0[i] as f64 * TWO64 + x.0[i - 1] as f64) * scale
}

/// The margin of the carry test's floating-point screen, far above its
/// error (below 1e-11, see [`TopFdSetup::carry`]).
const SCREEN_MARGIN: f64 = 1.0 / 1_048_576.0;

/// The field's constants for stepping its top prefixes.
pub(crate) struct TopFdSetup {
    b: u32,
    bias: u64,
    f0: u32,
    o: u32,
    s2: u32,
    s3: u32,
    cap: u32,
    w: u128,
    /// `b^o`, the step in `P`, and the farthest [`MAX_STEPS`] reach.
    h: u128,
    reach: u128,
    /// Words of the addends `DS`, `DC` and `DDC` at the field's last
    /// prefix (the largest values they take); each is below [`NW`].
    addend_words: [usize; 3],
    /// `2h²` and `6h³` as digit arrays, and the word each digit sits in.
    dds: Digits,
    dds_at: usize,
    dddc: Digits,
    dddc_at: usize,
    /// `h²` and `h³` (the seed's).
    h2: Digits,
    h3: Digits,
    /// `2δH` and `6δH²`.
    c2: W4,
    c3: W4,
    /// `b^i` (the base's power table) and `1/b^(i − 2)` as f64.
    pw: Vec<W4>,
    inv: Vec<f64>,
    /// `b^(2·f0)`, `b^(3·f0)`; `b²`, `b⁴`, `b⁸`; `b²` as f64.
    bz: [u128; 2],
    b2: u64,
    b4: u64,
    b8: u128,
    b2f: f64,
    /// `ndig(2^(l − 1))` at index `l` (`ndig(0) = 0` at 0).
    ndig_tab: [u8; 257],
    /// Division by `b^8`, `b^4` and `b` for the seed's digits.
    d8: Div64,
    d4: Div64,
    sb: SmallDiv,
    /// `b − 1` and `b^mid`: the join's `P mod (b − 1)` and `⌊P/h⌋ mod
    /// b^mid` step with `P`.
    m1: u32,
    bmid: u32,
}

/// The state at one prefix `P`: its digit arrays, the binary differences,
/// and `P mod (b − 1)` and `⌊P/h⌋ mod b^mid` (each grows by one a step,
/// since `h` is a power of `b`).
pub(crate) struct TopFd {
    p: u128,
    sq: Digits,
    ds: Digits,
    cu: Digits,
    dc: Digits,
    ddc: Digits,
    d2: W4,
    d3: W4,
    dd3: W4,
    pub(crate) pc: u32,
    pub(crate) dm: u32,
}

impl Default for TopFd {
    /// A state no prefix steps from: the first one seeds it.
    fn default() -> Self {
        Self {
            p: u128::MAX,
            sq: [0; NW],
            ds: [0; NW],
            cu: [0; NW],
            dc: [0; NW],
            ddc: [0; NW],
            d2: W4::default(),
            d3: W4::default(),
            dd3: W4::default(),
            pc: 0,
            dm: 0,
        }
    }
}

impl TopFdSetup {
    /// The constants for blocks of `w = b^f0`, prefixes stepped by `h = b^o`
    /// up to `phi`, and certificates from position `cap`; `bmid` as in
    /// [`TopFd`]. `None` if a digit array would not hold its
    /// values over the field, or the base has no room for the step's
    /// digits (`b <= 6`); the join then certifies every block by
    /// `Base::cert`.
    // b, w, h, o and f0 as in the module docs.
    #[allow(clippy::many_single_char_names)]
    pub(crate) fn try_new(
        base: &Base,
        f0: u32,
        o: u32,
        cap: u32,
        phi: u128,
        bmid: u32,
    ) -> Option<Self> {
        let b = base.b;
        if !(7..=64).contains(&b) || f0 == 0 || o == 0 {
            return None;
        }
        let bu = u128::from(b);
        let w = bu.checked_pow(f0)?;
        let h = bu.checked_pow(o)?;
        let hh = w.checked_mul(h)?;
        let bz = [bu.checked_pow(2 * f0)?, bu.checked_pow(3 * f0)?];
        let delta = w - 1;
        let pw = base.poww().to_vec();
        let ndig = |x: &W4| base.ndig_w(x) as usize;
        let words = |x: &W4| ndig(x).div_ceil(8);
        // Every array is increasing in P, so its value at phi bounds it.
        let p2 = W4::mul_u128_u128(phi, phi);
        let p3 = p2.mul_u128(phi);
        let h2 = W4::mul_u128_u128(h, h);
        let h3 = h2.mul_u128(h);
        let ds = W4::mul_u128_u128(2 * phi, h).add(&h2);
        let dc = p2
            .mul_u128(3 * h)
            .add(&W4::mul_u128_u128(3 * phi, h).mul_u128(h))
            .add(&h3);
        let ddc = W4::mul_u128_u128(6 * phi, h)
            .mul_u128(h)
            .add(&h3.mul_u128(6));
        let addend_words = [words(&ds), words(&dc), words(&ddc)];
        let (dds_at, dddc_at) = ((2 * o / 8) as usize, (3 * o / 8) as usize);
        // Reads of the arrays stay inside them: a^j < b^(s_j), so P^j has
        // at most s_j − j·f0 digits, and the certificate reads no higher.
        let fits = words(&p2) <= NW
            && words(&p3) <= NW
            && addend_words.iter().all(|&n| n < NW)
            && dddc_at + 1 < NW
            && base.s3 >= 3 * f0
            && base.s3 - 3 * f0 <= DIGITS
            && (base.s3 as usize) < pw.len()
            && hh.checked_mul(6 * delta).is_some();
        if !fits {
            return None;
        }
        let mut ndig_tab = [0u8; 257];
        for (l, n) in ndig_tab.iter_mut().enumerate().skip(1) {
            let mut x = W4::default();
            x.0[(l - 1) / 64] = 1 << ((l - 1) % 64);
            *n = u8::try_from(ndig(&x)).ok()?;
        }
        let inv = (0..pw.len())
            .map(|c| if c < 2 { 0.0 } else { 1.0 / w4_f64(&pw[c - 2]) })
            .collect();
        let b64 = u64::from(b);
        Some(Self {
            b,
            bias: 256 - b64,
            f0,
            o,
            s2: base.s2,
            s3: base.s3,
            cap,
            w,
            h,
            reach: h * u128::from(MAX_STEPS),
            addend_words,
            dds: digit_at(2, 2 * o),
            dds_at,
            dddc: digit_at(6, 3 * o),
            dddc_at,
            h2: digit_at(1, 2 * o),
            h3: digit_at(1, 3 * o),
            c2: W4::mul_u128_u128(2 * delta, hh),
            c3: W4::mul_u128_u128(6 * delta * hh, hh),
            pw,
            inv,
            bz,
            b2: b64 * b64,
            b4: b64.pow(4),
            b8: bu.pow(8),
            b2f: f64::from(b * b),
            ndig_tab,
            d8: Div64::new(b64.pow(8)),
            d4: Div64::new(b64.pow(4)),
            sb: SmallDiv::new(b64),
            m1: b - 1,
            bmid,
        })
    }

    /// The digits of `x` (below `b^48`): eight at a time by a division by
    /// `b^8`, each eight split by `b^4` and then digit by digit with a
    /// reciprocal multiply.
    // Remainders below b^8 <= 2^48 and digits below b.
    #[allow(clippy::cast_possible_truncation)]
    fn to_digits(&self, mut x: W4) -> Digits {
        let mut out = [0u64; NW];
        let mut i = 0;
        while x != W4::default() {
            let chunk = self.d8.divrem_w4(&mut x);
            let (hi, mut lo) = self.d4.divrem_u128(u128::from(chunk));
            let mut hi = hi as u64;
            let mut word = 0u64;
            for j in 0..4 {
                let (q, r) = self.sb.divrem(lo);
                word |= r << (8 * j);
                lo = q;
                let (q, r) = self.sb.divrem(hi);
                word |= r << (8 * (j + 4));
                hi = q;
            }
            out[i] = word;
            i += 1;
        }
        out
    }

    /// The state at `p` (an unclipped block of the field) from scratch.
    // The join's residues: below b − 1 and b^mid.
    #[allow(clippy::cast_possible_truncation)]
    fn seed(&self, st: &mut TopFd, p: u128) {
        let bias = self.bias;
        let (o, h) = (self.o, self.h);
        let pd = self.to_digits(w4(p));
        let p2 = W4::mul_u128_u128(p, p);
        st.sq = self.to_digits(p2);
        st.cu = self.to_digits(p2.mul_u128(p));
        let two_p = sum(&pd, &pd, bias);
        let three_p = sum(&two_p, &pd, bias);
        let six_p = sum(&three_p, &three_p, bias);
        let three_sq = sum(&sum(&st.sq, &st.sq, bias), &st.sq, bias);
        // DS = 2P·h + h², DC = 3P²·h + 3P·h² + h³, DDC = 6P·h² + 6h³.
        st.ds = sum(&shl_digits(&two_p, o), &self.h2, bias);
        st.dc = sum(
            &sum(
                &shl_digits(&three_sq, o),
                &shl_digits(&three_p, 2 * o),
                bias,
            ),
            &self.h3,
            bias,
        );
        st.ddc = sum(&shl_digits(&six_p, 2 * o), &self.dddc, bias);
        let (a, dl, hh) = (p * self.w, self.w - 1, self.w * h);
        st.d2 = W4::mul_u128_u128(dl, 2 * a + dl);
        st.d3 = W4::mul_u128_u128(a, a)
            .mul_u128(3)
            .add(&W4::mul_u128_u128(3 * a + dl, dl))
            .mul_u128(dl);
        st.dd3 = W4::mul_u128_u128(3 * dl * hh, 2 * a + hh + dl);
        st.pc = (p % u128::from(self.m1)) as u32;
        st.dm = ((p / h) % u128::from(self.bmid)) as u32;
        st.p = p;
    }

    /// `P → P + h`: five digit-array additions and three binary ones.
    #[inline(always)]
    fn step(&self, st: &mut TopFd) {
        let bias = self.bias;
        let [ds, dc, ddc] = self.addend_words;
        add_digits(&mut st.sq, &st.ds, 0, ds, bias);
        add_digits(&mut st.ds, &self.dds, self.dds_at, self.dds_at + 1, bias);
        add_digits(&mut st.cu, &st.dc, 0, dc, bias);
        add_digits(&mut st.dc, &st.ddc, 0, ddc, bias);
        add_digits(
            &mut st.ddc,
            &self.dddc,
            self.dddc_at,
            self.dddc_at + 1,
            bias,
        );
        st.d2 = st.d2.add(&self.c2);
        st.d3 = st.d3.add(&st.dd3);
        st.dd3 = st.dd3.add(&self.c3);
        st.p += self.h;
        st.pc = if st.pc + 1 == self.m1 { 0 } else { st.pc + 1 };
        st.dm = if st.dm + 1 == self.bmid { 0 } else { st.dm + 1 };
    }

    /// Bring `st` to `p`, an unclipped block of the field: by steps if `p`
    /// is at most [`MAX_STEPS`] steps above its prefix, else by a seed.
    #[inline]
    pub(crate) fn advance(&self, st: &mut TopFd, p: u128) {
        if p > st.p && p - st.p <= self.reach {
            while st.p < p {
                self.step(st);
            }
            if st.p == p {
                return;
            }
        }
        self.seed(st, p);
    }

    /// `ndig(d)`: smallest `i` with `b^i > d`. `d` lies in `[2^(l − 1),
    /// 2^l)` for its bit length `l`, and `2 < b`, so it has `ndig(2^(l − 1))`
    /// digits or one more.
    #[inline(always)]
    fn ndig(&self, d: &W4) -> u32 {
        let c = u32::from(self.ndig_tab[bit_len(d)]);
        c + u32::from(!d.lt(&self.pw[c as usize]))
    }

    /// `x mod b^k` for the digit array `x`.
    fn low_value(&self, x: &Digits, k: u32) -> W4 {
        let b = u64::from(self.b);
        let (full, part) = ((k / 8) as usize, k % 8);
        let mut acc = W4::default();
        if part > 0 {
            let top = x[full] & ((1u64 << (8 * part)) - 1);
            acc = w4(u128::from(word_value(top, b, self.b2, self.b4)));
        }
        for i in (0..full).rev() {
            let v = w4(u128::from(word_value(x[i], b, self.b2, self.b4)));
            acc = acc.mul_u128(self.b8).add(&v);
        }
        acc
    }

    /// `Base::cert`'s carry test, `(a^j mod b^c0) + d >= b^c0`, for `a^j =
    /// X·b^z` with `X` the digit array `x` and `d < b^c0`.
    ///
    /// With `k = c0 − z >= 2` and `R = X mod b^k`, divide by `b^(c0 − 2)`:
    /// the test is `R/b^(k − 2) + d/b^(c0 − 2) >= b²`. The first term lies in
    /// `[r, r + 1)` for `r` the top two digits of `R`, and the second is
    /// below `b²` and computed in f64 within 1e-11 (three roundings of
    /// relative error `2^-51` at most, on a value below `2^12`). So outside
    /// [`SCREEN_MARGIN`] of the two bounds the answer is certain; inside it
    /// (about `1/b²` of the calls), and for `k < 2`, `R` is put together
    /// from the digits and the sum compared exactly.
    // x, d, z, k, r as above.
    #[allow(clippy::many_single_char_names)]
    #[inline(always)]
    fn carry(&self, x: &Digits, d: &W4, c0: u32, z: u32, bz: u128) -> bool {
        if c0 >= z + 2 {
            let k = c0 - z;
            let r = f64::from(digit(x, k - 1)) * f64::from(self.b) + f64::from(digit(x, k - 2));
            let df = w4_f64(d) * self.inv[c0 as usize];
            if r + df >= self.b2f + SCREEN_MARGIN {
                return true;
            }
            if r + 1.0 + df <= self.b2f - SCREEN_MARGIN {
                return false;
            }
        }
        let rem = if c0 > z {
            self.low_value(x, c0 - z).mul_u128(bz)
        } else {
            W4::default()
        };
        !rem.add(d).lt(&self.pw[c0 as usize])
    }

    /// The certificate of the block of `st`'s prefix (unclipped): the same
    /// as `Base::cert_parts(a, a + w − 1, cap)`, read from the stepped
    /// digits. The digit of `a^j` at position `pos` is digit `pos − j·f0`
    /// of `P^j` (zero below).
    #[inline]
    pub(crate) fn cert(&self, st: &TopFd) -> Option<Cert> {
        // A digit, below b <= 64.
        #[allow(clippy::cast_possible_truncation)]
        let top = (self.b - 1) as u8;
        let mut mask = 0u64;
        let mut doms = [0u64; 2];
        let mut from = [self.s2, self.s3];
        let powers = [
            (&st.sq, &st.d2, self.s2, 2 * self.f0, self.bz[0]),
            (&st.cu, &st.d3, self.s3, 3 * self.f0, self.bz[1]),
        ];
        for ((dom, first), (x, d, sp, z, bz)) in doms.iter_mut().zip(&mut from).zip(powers) {
            let c0 = self.cap.max(self.ndig(d));
            if c0 >= sp {
                continue;
            }
            let at = |pos: u32| if pos < z { 0 } else { digit(x, pos - z) };
            let mut pos = c0;
            if self.carry(x, d, c0, z, bz) {
                // As in `Base::cert`: the run of b − 1 ends below sp.
                while at(pos) == top {
                    pos += 1;
                }
                let g = at(pos);
                *dom = digit_range_mask(g, g + 1);
                pos += 1;
            }
            *first = pos;
            while pos < sp {
                let bit = 1u64 << at(pos);
                if mask & bit != 0 {
                    return None;
                }
                mask |= bit;
                pos += 1;
            }
        }
        close_certificate(mask, doms).map(|closed| Cert {
            mask: closed,
            common: mask,
            from,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base_range::get_base_range_u128;

    /// A small LCG: the test's samples, reproducible.
    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            self.0 >> 16
        }

        /// Uniform enough below `n > 0`.
        fn below(&mut self, n: u128) -> u128 {
            ((u128::from(self.next()) << 48) ^ u128::from(self.next())) % n
        }
    }

    /// Bases 10-64: random fields inside each band, cut into blocks of `b^f0`
    /// for several `f0` and stepped by `b^o` for `o` = 1-3, in runs from
    /// random starts with random gaps (most of them steps, some past
    /// [`MAX_STEPS`], some backwards), from the field's first prefix and to
    /// its last. Clipped blocks are left out and leave the state alone, as
    /// in the join. Every other block's certificate equals `Base::cert`'s,
    /// and its `P mod (b − 1)` and `⌊P/h⌋ mod b²` are `P`'s.
    #[test]
    #[allow(clippy::many_single_char_names)] // b, s, e, l, w, o, p as above
    fn stepped_certificates_equal_the_divided_ones() {
        let mut rng = Lcg(0x5DEE_CE66_D1CE_4E5B);
        let (mut setups, mut certs, mut kept, mut clipped, mut seeds) =
            (0u32, 0u64, 0u64, 0u64, 0u64);
        for b in 10..=64u32 {
            let Some(band) = get_base_range_u128(b).expect("base range") else {
                continue;
            };
            let mut fields = 0;
            for _ in 0..40 {
                if fields == 4 {
                    break;
                }
                let s = band.start() + rng.below(band.size());
                let e = s + 1 + rng.below(band.end() - s);
                let Some(base) = Base::try_new(b, s, e - 1) else {
                    continue;
                };
                fields += 1;
                let l = base.l;
                for f0 in 1..l {
                    // The join's blocks are below 2^32.
                    let w = base.powu(f0);
                    if w >= 1 << 32 {
                        break;
                    }
                    for o in 1..=3u32 {
                        let cap = f0 + o;
                        if cap >= l {
                            continue;
                        }
                        let (plo, phi) = (s / w, (e - 1) / w);
                        let Some(fd) = TopFdSetup::try_new(&base, f0, o, cap, phi, b * b) else {
                            continue;
                        };
                        setups += 1;
                        let h = base.powu(o);
                        for run in 0..8 {
                            let mut st = TopFd::default();
                            let span = phi - plo + 1;
                            let mut p = match run {
                                0 => plo,
                                1 => phi.saturating_sub(40 * h).max(plo),
                                _ => plo + rng.below(span),
                            };
                            for _ in 0..200 {
                                if p > phi {
                                    break;
                                }
                                let lo = p * w;
                                let (a, z) = (lo.max(s), (lo + w - 1).min(e - 1));
                                if a == lo && z == lo + w - 1 {
                                    let before = st.p;
                                    fd.advance(&mut st, p);
                                    seeds += u64::from(!(p > before && p - before <= fd.reach));
                                    let want = base.cert_parts(a, z, cap);
                                    assert_eq!(
                                        fd.cert(&st),
                                        want,
                                        "b{b} [{s}, {e}) f0 {f0} o {o} P {p}"
                                    );
                                    assert_eq!(
                                        st.pc,
                                        u32::try_from(p % u128::from(b - 1)).unwrap()
                                    );
                                    assert_eq!(
                                        st.dm,
                                        u32::try_from(p / h % u128::from(b * b)).unwrap()
                                    );
                                    certs += 1;
                                    kept += u64::from(want.is_some());
                                } else {
                                    clipped += 1;
                                }
                                p = match rng.next() % 32 {
                                    0 => p.saturating_sub(h * rng.below(8)),
                                    1..=3 => p + h * (5 + rng.below(20)),
                                    4..=11 => p + h * (2 + rng.below(3)),
                                    _ => p + h,
                                };
                            }
                        }
                    }
                }
            }
        }
        eprintln!(
            "{setups} setups: {certs} blocks certified ({kept} kept, {seeds} seeds), {clipped} clipped left out"
        );
        assert!(setups > 1500 && certs > 1_000_000 && clipped > 1000 && kept > 100_000);
    }

    /// A carry through words of b − 1: into word 0, through the words above
    /// it (a carry in over eight digit sums of b − 1), and past the carry
    /// word. The certificates' runs rarely reach either.
    #[test]
    fn a_carry_runs_through_words_of_b_minus_1() {
        for b in 7..=64u32 {
            let (bias, top) = (256 - u64::from(b), u64::from(b - 1));
            for hi in 1..NW - 1 {
                let mut x = [0u64; NW];
                x[..=hi].fill(top * LO);
                x[hi + 1] = top | (top << 8);
                let mut one = [0u64; NW];
                one[0] = 1;
                add_digits(&mut x, &one, 0, hi, bias);
                assert!(x[..=hi].iter().all(|&w| w == 0), "b{b} hi {hi}");
                assert_eq!(x[hi + 1], 1 << 16, "b{b} hi {hi}");
            }
        }
    }
}
