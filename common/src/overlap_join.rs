//! The overlap join: a nice-only search organised as a join between a list of
//! certified top prefixes and a list of certified bottom residues, instead of
//! a walk over every stride candidate of every MSD range.
//!
//! This module is the CPU half: the exact arithmetic (certificates, the two
//! lists), the per-field parameters, and a reference implementation of the
//! whole join that the GPU stage in `cubecl_join` is tested against. The
//! design, the CPU prototype this is ported from and the GPU kernels are
//! Dan Stoyell's (wasabipesto/nice#177; `overlap-join/` in
//! `danstoyell/nice_numbers_research` at `ba53f5c`).
//!
//! # The idea
//!
//! For an `L`-digit candidate `n` and parameters `(t, k, p)` with
//! `t + k > L`:
//!
//! - **Top list:** every prefix `P` of the top `t` digits of `n` whose output
//!   digits of `n²` and `n³` that are constant over `P`'s interval (clipped
//!   to the field), at positions `≥ k`, are all distinct. Its certificate is
//!   the mask of those digits.
//! - **Bottom list:** every residue `R = n mod b^k` whose low `k` digits of
//!   `n²` and of `n³` (which `R` determines exactly) are all distinct; its
//!   certificate is their mask.
//! - The two lists share `o = t + k − L` digits of `n` (positions
//!   `f0 = L − t` up to `k − 1`). Every `n` of the field is exactly one
//!   `(P, R)` pair that agrees on them. The lowest `p` shared digits are the
//!   *partition* (both lists are built per partition value), the rest are a
//!   hash key, and so is the digit-sum class: `n = P·b^f0 + R_low ≡ P + R_low
//!   (mod b − 1)` must be a root of `n² + n³ ≡ b(b−1)/2`, so each top probes
//!   one bucket per root.
//! - Each matched pair costs one 64-bit AND of the two masks, which cover
//!   disjoint output positions. Pairs that pass are fully checked.
//!
//! # Soundness
//!
//! A nice `n` has distinct digits on every subset of its output positions.
//! So its prefix is in the top list, its residue is in the bottom list, its
//! class is a root, its pair agrees on the shared digits and is matched, and
//! the AND of the two masks is zero. Nothing else is assumed: this is the
//! client's own MSD, LSD, residue and cross-end reasoning, enumerated as a
//! join.

use crate::FieldSize;
use crate::client_process::get_is_nice;
use anyhow::{Result, anyhow, ensure};
use std::sync::OnceLock;
use std::time::Instant;

/// Smallest base the join is used for. Below this the stride pipeline is as
/// fast or faster and long finished anyway.
pub const JOIN_MIN_BASE: u32 = 40;
/// Largest base: certificates are `u64` digit masks.
pub const JOIN_MAX_BASE: u32 = 64;
/// Smallest field the join is used for. The join pays a per-field cost that
/// does not shrink with the field (the bottom list is rebuilt for every
/// partition value), so on small fields — every benchmark window, for
/// example — the stride pipeline stays faster. Production fields at the
/// frontier are 1e14.
pub const JOIN_MIN_FIELD_SIZE: u128 = 10_000_000_000_000;

/// A nice-only field's route as `NICE_JOIN_ROUTE` sets it, for testing and
/// A/B runs while the join settles (to be removed once it has): `join` sends
/// every field the join can take to it, whatever its size (the other limits
/// of [`join_verdict`] and the device's still hold); `stride` keeps every
/// field on the stride path. Unset, `auto`, or anything else decides field
/// by field. It applies to the CPU and the GPU, and to the benchmark.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RouteOverride {
    Auto,
    Join,
    Stride,
}

impl RouteOverride {
    fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "" | "auto" => Some(Self::Auto),
            "join" => Some(Self::Join),
            "stride" => Some(Self::Stride),
            _ => None,
        }
    }
}

/// The process's [`RouteOverride`], read from `NICE_JOIN_ROUTE` once (and
/// logged once if it overrides anything).
#[must_use]
pub fn route_override() -> RouteOverride {
    static ROUTE: OnceLock<RouteOverride> = OnceLock::new();
    *ROUTE.get_or_init(|| {
        let Ok(value) = std::env::var("NICE_JOIN_ROUTE") else {
            return RouteOverride::Auto;
        };
        let route = RouteOverride::parse(&value).unwrap_or_else(|| {
            log::warn!("NICE_JOIN_ROUTE={value} is not auto, join or stride; ignoring it");
            RouteOverride::Auto
        });
        if route != RouteOverride::Auto {
            log::warn!(
                "NICE_JOIN_ROUTE={value}: nice-only fields take the {} path where they can",
                if route == RouteOverride::Join {
                    "overlap join"
                } else {
                    "stride"
                }
            );
        }
        route
    })
}

/// The join's shape for one field: `t` top digits, `k` bottom digits,
/// `p` partition digits.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct JoinParams {
    pub t: u32,
    pub k: u32,
    pub p: u32,
}

/// Digits of `x` in base `b` (`0` has none).
#[must_use]
pub fn ndigits(mut x: u128, b: u32) -> u32 {
    let mut n = 0;
    while x > 0 {
        x /= u128::from(b);
        n += 1;
    }
    n
}

impl JoinParams {
    /// The parameters for `base` at digit length `l`, if the GPU stage
    /// supports that combination: the overlap `o = t + k − l` must leave at
    /// most one key digit above the partition digits, a top block `b^f0` and
    /// the prefilter's `P mod b^(k+2−f0)` must fit `u32`, a bottom residue
    /// `b^k` fits 40 bits, and the cube of the largest residue fits `u128`
    /// (the reference arithmetic).
    #[must_use]
    pub fn for_length(base: u32, l: u32) -> Option<Self> {
        if !(JOIN_MIN_BASE..=JOIN_MAX_BASE).contains(&base) || l < 7 {
            return None;
        }
        let jp = JoinParams {
            t: l - 3,
            k: 6,
            p: 2,
        };
        jp.supported(base, l).then_some(jp)
    }

    /// Whether the GPU stage can run these parameters at digit length `l`.
    #[must_use]
    // t, k, p, o and f0 are the module docs' names.
    #[allow(clippy::many_single_char_names)]
    pub fn supported(&self, base: u32, l: u32) -> bool {
        let (t, k, p) = (self.t, self.k, self.p);
        if base > JOIN_MAX_BASE || t + k <= l || t > l || k >= l {
            return false;
        }
        let f0 = l - t;
        let o = t + k - l;
        let b = u128::from(base);
        p <= o
            && o - p <= 1
            && b.pow(k) < 1 << 40
            && b.pow(f0) < 1 << 32
            && b.checked_pow(3 * k).is_some()
            && k > f0
    }
}

/// Why a nice-only field takes the stride path instead of the overlap join.
/// The join's own reasons come from [`join_verdict`]; the GPU route adds
/// the device's and the backend's (`crate::gpu_route::begin_niceonly`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StrideReason {
    /// The base is outside [`JOIN_MIN_BASE`]..=[`JOIN_MAX_BASE`], or the
    /// join has no parameters for its digit length.
    Base,
    /// The field is smaller than [`JOIN_MIN_FIELD_SIZE`].
    BelowMinSize,
    /// `n`, `n²` or `n³` changes digit length inside the field.
    LengthChange,
    /// `n` reaches 2^96, past the device stage's width.
    TooWide,
    /// The GPU cannot hold one partition of a slice of the field.
    DeviceTooSmall,
    /// The backend has no overlap join (hand-CUDA).
    NoJoin,
    /// The device has no plane (subgroup) operations, which the join's
    /// kernels need (some wgpu adapters, such as DX12 without DXC).
    NoPlaneOps,
    /// The join could not set the field up (not expected at bases 40-64).
    Setup,
    /// `NICE_JOIN_ROUTE=stride` ([`RouteOverride`]).
    Forced,
}

impl StrideReason {
    /// A short label, as logged and as sent in telemetry.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Base => "base outside the join's range",
            Self::BelowMinSize => "field below the join's minimum size",
            Self::LengthChange => "digit length changes inside the field",
            Self::TooWide => "n too wide for the device stage",
            Self::DeviceTooSmall => "device cannot hold one partition",
            Self::NoJoin => "backend has no overlap join",
            Self::NoPlaneOps => "device has no subgroup operations",
            Self::Setup => "the join could not set the field up",
            Self::Forced => "forced by NICE_JOIN_ROUTE=stride",
        }
    }
}

/// The join parameters for a field, or why the field takes the stride
/// path: outside [`JOIN_MIN_BASE`]..=[`JOIN_MAX_BASE`], smaller than
/// [`JOIN_MIN_FIELD_SIZE`], past the device stage's 96-bit `n`, or crossing a
/// digit-length boundary of `n`, `n²` or `n³`; `NICE_JOIN_ROUTE` can force
/// either path ([`RouteOverride`]).
///
/// # Errors
/// The reason the field takes the stride path.
pub fn join_verdict(base: u32, range: &FieldSize) -> Result<JoinParams, StrideReason> {
    join_verdict_routed(base, range, route_override())
}

/// [`join_verdict`] under the route override `route`.
pub(crate) fn join_verdict_routed(
    base: u32,
    range: &FieldSize,
    route: RouteOverride,
) -> Result<JoinParams, StrideReason> {
    if route == RouteOverride::Stride {
        return Err(StrideReason::Forced);
    }
    if !(JOIN_MIN_BASE..=JOIN_MAX_BASE).contains(&base) {
        return Err(StrideReason::Base);
    }
    if route != RouteOverride::Join && range.size() < JOIN_MIN_FIELD_SIZE {
        return Err(StrideReason::BelowMinSize);
    }
    if range.last() >= 1u128 << 96 {
        return Err(StrideReason::TooWide);
    }
    let l = ndigits(range.last(), base);
    let jp = JoinParams::for_length(base, l).ok_or(StrideReason::Base)?;
    // Lengths of n, n² and n³ must be constant over the field.
    Base::try_new(base, range.first(), range.last()).ok_or(StrideReason::LengthChange)?;
    Ok(jp)
}

/// What the overlap join did with one field, as sent in telemetry (the
/// GPU's in `gpu_niceonly::NiceonlyStats`, the CPU's by the client).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct JoinTelemetry {
    /// Host setup of every slice (top layer and bottom list), and the
    /// partitions' run time: on the GPU the device's wall time, on the CPU
    /// the partitions' wall time on the client's threads.
    pub setup_secs: f64,
    pub run_secs: f64,
    /// Slices, and partitions per slice.
    pub slices: usize,
    pub partitions: usize,
    /// GPU: partitions per launch planned, and the fewest any launch used
    /// (both 0 on the CPU).
    pub slots: usize,
    pub min_slots: usize,
    /// GPU: partitions re-run after their batch overflowed its list, and
    /// halvings of a partition's top layer.
    pub retried_partitions: usize,
    pub splits: u32,
    /// GPU: layouts the device did not run, each followed by one of at most
    /// half the size (normally none).
    pub refused: u32,
    /// Pairs that passed the join's AND, and of those the prefilter's
    /// survivors, which the full check read.
    pub survivors: u64,
    pub checked: u64,
}

impl JoinTelemetry {
    /// As the `join` object of the telemetry JSON.
    #[must_use]
    pub fn json(&self) -> serde_json::Value {
        serde_json::json!({
            "setup_secs": self.setup_secs,
            "run_secs": self.run_secs,
            "slices": self.slices,
            "partitions": self.partitions,
            "slots": self.slots,
            "min_slots": self.min_slots,
            "retried_partitions": self.retried_partitions,
            "splits": self.splits,
            "refused_layouts": self.refused,
            "survivors": self.survivors,
            "checked": self.checked,
        })
    }
}

/// [`join_verdict`] without the reason: the join parameters for a field, or
/// `None` if it takes the stride path.
#[must_use]
pub fn join_params_for(base: u32, range: &FieldSize) -> Option<JoinParams> {
    join_verdict(base, range).ok()
}

/// How many numbers one top-layer prefix covers: `b^(L − t + p)`, the
/// granularity a field is sliced at ([`join_slices`]).
#[must_use]
pub fn prefix_block(base: u32, l: u32, jp: JoinParams) -> u128 {
    u128::from(base).pow(l - (jp.t - jp.p))
}

/// `range` cut into consecutive slices of at most `max_prefixes` top-layer
/// prefixes each, at multiples of `block` ([`prefix_block`]); one slice if
/// the field has no more prefixes than that.
///
/// Every structure the join builds per field grows with its top layer: the
/// host's (`FieldSetup::tlay`), the device's top buffers, and the
/// survivors a partition yields. A field is processed slice by slice, so
/// all of them are bounded by `max_prefixes` whatever the field's size, at
/// the cost of the per-field fixed work (the bottom side, every partition)
/// once per slice.
#[must_use]
pub fn join_slices(range: &FieldSize, block: u128, max_prefixes: u128) -> Vec<FieldSize> {
    let step = block.saturating_mul(max_prefixes.max(1));
    let mut out = Vec::new();
    let mut s = range.start();
    while s < range.end() {
        let e = (s / block * block).saturating_add(step).min(range.end());
        out.push(FieldSize::new(s, e));
        s = e;
    }
    out
}

/// Four little-endian `u64` words: enough for `n³` with `n < 2^85`.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct W4(pub [u64; 4]);

// Word arithmetic: every `as u64` keeps the low word on purpose, and the
// loops carry across parallel word arrays by index.
#[allow(clippy::cast_possible_truncation, clippy::needless_range_loop)]
impl W4 {
    #[inline]
    #[must_use]
    pub fn mul_u128_u128(a: u128, b: u128) -> Self {
        let (a0, a1, b0, b1) = (u128::from(a as u64), a >> 64, u128::from(b as u64), b >> 64);
        let p00 = a0 * b0;
        let p01 = a0 * b1;
        let p10 = a1 * b0;
        let p11 = a1 * b1;
        let mid = (p00 >> 64) + u128::from(p01 as u64) + u128::from(p10 as u64);
        let hi = (mid >> 64) + (p01 >> 64) + (p10 >> 64) + u128::from(p11 as u64);
        W4([
            p00 as u64,
            mid as u64,
            hi as u64,
            ((hi >> 64) + (p11 >> 64)) as u64,
        ])
    }

    #[inline]
    #[must_use]
    pub fn mul_u128(&self, b: u128) -> Self {
        let (b0, b1) = (u128::from(b as u64), b >> 64);
        let mut r = [0u64; 4];
        let mut c: u128 = 0;
        for i in 0..4 {
            c += u128::from(self.0[i]) * b0 + u128::from(r[i]);
            r[i] = c as u64;
            c >>= 64;
        }
        c = 0;
        for i in 0..3 {
            c += u128::from(self.0[i]) * b1 + u128::from(r[i + 1]);
            r[i + 1] = c as u64;
            c >>= 64;
        }
        W4(r)
    }

    #[inline]
    #[must_use]
    pub fn sub(&self, o: &W4) -> W4 {
        let mut r = [0u64; 4];
        let mut borrow = 0u64;
        for i in 0..4 {
            let (d1, b1) = self.0[i].overflowing_sub(o.0[i]);
            let (d2, b2) = d1.overflowing_sub(borrow);
            r[i] = d2;
            borrow = u64::from(b1 | b2);
        }
        W4(r)
    }

    #[inline]
    #[must_use]
    pub fn lt(&self, o: &W4) -> bool {
        for i in (0..4).rev() {
            if self.0[i] != o.0[i] {
                return self.0[i] < o.0[i];
            }
        }
        false
    }

    #[inline]
    #[must_use]
    pub fn to_u128(&self) -> u128 {
        debug_assert!(self.0[2] == 0 && self.0[3] == 0);
        (u128::from(self.0[1]) << 64) | u128::from(self.0[0])
    }
}

/// Division by an invariant 64-bit divisor with a precomputed reciprocal
/// (N. Möller and T. Granlund, "Improved division by invariant integers",
/// IEEE Trans. Computers 60 (2011), Algorithm 4).
#[derive(Clone, Copy, Debug)]
pub struct Div64 {
    dn: u64,
    shift: u32,
    v: u64,
}

// The same: `as u64` takes the low word of a 128-bit product or quotient.
#[allow(clippy::cast_possible_truncation)]
impl Div64 {
    #[must_use]
    pub fn new(d: u64) -> Self {
        let shift = d.leading_zeros();
        let dn = d << shift;
        let v = (u128::MAX / u128::from(dn) - (1u128 << 64)) as u64;
        Div64 { dn, shift, v }
    }

    /// (u1:u0) / dn with u1 < dn -> (q, r)
    #[inline]
    fn div2by1(&self, u1: u64, u0: u64) -> (u64, u64) {
        let q = (u128::from(self.v) * u128::from(u1))
            .wrapping_add((u128::from(u1) << 64) | u128::from(u0));
        let mut q1 = ((q >> 64) as u64).wrapping_add(1);
        let q0 = q as u64;
        let mut r = u0.wrapping_sub(q1.wrapping_mul(self.dn));
        if r > q0 {
            q1 = q1.wrapping_sub(1);
            r = r.wrapping_add(self.dn);
        }
        if r >= self.dn {
            q1 += 1;
            r -= self.dn;
        }
        (q1, r)
    }

    /// x /= d in place, returns x mod d.
    #[inline]
    pub fn divrem_w4(&self, x: &mut W4) -> u64 {
        let mut i = 4;
        while i > 0 && x.0[i - 1] == 0 {
            i -= 1;
        }
        if i == 0 {
            return 0;
        }
        let sh = self.shift;
        let mut r: u64 = if sh == 0 { 0 } else { x.0[i - 1] >> (64 - sh) };
        while i > 0 {
            i -= 1;
            let lo = if sh == 0 {
                x.0[i]
            } else {
                (x.0[i] << sh) | if i > 0 { x.0[i - 1] >> (64 - sh) } else { 0 }
            };
            let (q, rr) = self.div2by1(r, lo);
            x.0[i] = q;
            r = rr;
        }
        r >> sh
    }

    #[inline]
    #[must_use]
    pub fn divrem_u128(&self, x: u128) -> (u128, u64) {
        let mut w = W4([x as u64, (x >> 64) as u64, 0, 0]);
        let r = self.divrem_w4(&mut w);
        (w.to_u128(), r)
    }
}

/// Per-base arithmetic for one field: digit lengths, digit-sum roots and
/// power tables.
#[derive(Clone)]
pub struct Base {
    pub b: u32,
    /// Digits of every n in the field.
    pub l: u32,
    /// Digits of n² and n³ (constant over the field).
    pub s2: u32,
    pub s3: u32,
    pub m1: u32,
    /// Residues r mod (b − 1) with r² + r³ ≡ b(b − 1)/2.
    pub roots: Vec<u32>,
    pow: Vec<u128>,
    poww: Vec<W4>,
    chd: u32,
    dpow: Vec<Div64>,
}

/// Digits the certificate's buffers hold (`Base::cert`): the cube of any
/// field the join takes has at most 39, at bases 40-64.
const DIGIT_BUF: usize = 48;

impl Base {
    /// `lo..=hi` must share the digit length of n, n² and n³ (true inside
    /// any field of a nice band), and n³ must have at most [`DIGIT_BUF`]
    /// digits; `None` otherwise.
    #[must_use]
    pub fn try_new(b: u32, lo: u128, hi: u128) -> Option<Self> {
        if !(3..=JOIN_MAX_BASE).contains(&b) || lo == 0 || lo > hi {
            return None;
        }
        let l = ndigits(hi, b);
        if l != ndigits(lo, b) {
            return None;
        }
        let m1 = b - 1;
        let tgt = (u64::from(b) * (u64::from(b) - 1) / 2) % u64::from(m1);
        let roots = (0..m1)
            .filter(|&r| {
                let r = u64::from(r);
                (r * r + r * r * r) % u64::from(m1) == tgt
            })
            .collect();
        let mut pow = vec![1u128];
        let mut x = 1u128;
        while pow.len() < 40 {
            match x.checked_mul(u128::from(b)) {
                Some(y) => {
                    pow.push(y);
                    x = y;
                }
                None => break,
            }
        }
        let mut poww = vec![W4([1, 0, 0, 0])];
        let mut last = poww[0];
        for _ in 0..60 {
            if last.0[3] >= (u64::MAX / u64::from(b)) {
                break;
            }
            last = last.mul_u128(u128::from(b));
            poww.push(last);
        }
        let mut pow64 = vec![1u64];
        let mut x64 = 1u64;
        while let Some(y) = x64.checked_mul(u64::from(b)) {
            pow64.push(y);
            x64 = y;
        }
        // Both tables hold at most 64 entries.
        let chd = u32::try_from(pow64.len() - 1).ok()?;
        let nw = u32::try_from(poww.len()).ok()?;
        let digits_w = |x: &W4| {
            let mut n = 0u32;
            while n + 1 < nw && !x.lt(&poww[n as usize]) {
                n += 1;
            }
            n
        };
        // n³ must fit the 256-bit words.
        if ndigits(hi, 2) * 3 > 255 {
            return None;
        }
        let s2 = digits_w(&W4::mul_u128_u128(lo, lo));
        let s3 = digits_w(&W4::mul_u128_u128(lo, lo).mul_u128(lo));
        let hs2 = digits_w(&W4::mul_u128_u128(hi, hi));
        let hs3 = digits_w(&W4::mul_u128_u128(hi, hi).mul_u128(hi));
        if s2 != hs2 || s3 != hs3 || s3 as usize > DIGIT_BUF {
            return None;
        }
        let dpow = pow64.iter().map(|&x| Div64::new(x)).collect();
        Some(Base {
            b,
            l,
            s2,
            s3,
            m1,
            roots,
            pow,
            poww,
            chd,
            dpow,
        })
    }

    /// b^i (i < 40, and b^i < 2^128).
    #[inline]
    #[must_use]
    pub fn powu(&self, i: u32) -> u128 {
        self.pow[i as usize]
    }

    /// b^i as four little-endian u64 words (i < the 256-bit power table).
    #[must_use]
    pub fn poww_words(&self, i: u32) -> [u64; 4] {
        self.poww[i as usize].0
    }

    /// Smallest i with b^i > x.
    #[inline]
    fn ndig_w(&self, x: &W4) -> u32 {
        let (mut lo, mut hi) = (0usize, self.poww.len() - 1);
        while lo < hi {
            let mid = usize::midpoint(lo, hi);
            if x.lt(&self.poww[mid]) {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        // An index into the power table (< 64 entries).
        u32::try_from(lo).unwrap_or(u32::MAX)
    }

    /// floor(x / b^c).
    #[inline]
    fn shift_down(&self, mut x: W4, mut c: u32) -> W4 {
        while c >= self.chd {
            self.dpow[self.chd as usize].divrem_w4(&mut x);
            c -= self.chd;
        }
        if c > 0 {
            self.dpow[c as usize].divrem_w4(&mut x);
        }
        x
    }

    /// Digits of x (least significant first), exactly `len` of them
    /// (x < b^len).
    // Digits are below b <= 64, so `as u8` is exact; x fits u64 by then.
    #[allow(clippy::cast_possible_truncation, clippy::many_single_char_names)]
    #[inline]
    fn digits_w(&self, mut x: W4, len: u32, out: &mut [u8; DIGIT_BUF]) {
        let b = u64::from(self.b);
        let mut n = 0usize;
        while x.0[2] != 0 || x.0[3] != 0 {
            let mut r = self.dpow[self.chd as usize].divrem_w4(&mut x);
            for _ in 0..self.chd {
                out[n] = (r % b) as u8;
                r /= b;
                n += 1;
            }
        }
        let mut q = x.to_u128();
        while q >> 64 != 0 {
            let (hi, r0) = self.dpow[self.chd as usize].divrem_u128(q);
            let mut r = r0;
            for _ in 0..self.chd {
                out[n] = (r % b) as u8;
                r /= b;
                n += 1;
            }
            q = hi;
        }
        let mut r = q as u64;
        while n < len as usize {
            out[n] = (r % b) as u8;
            r /= b;
            n += 1;
        }
    }

    /// Certificate of `[a, e]`: the mask of the output digits of n² and n³
    /// at positions `>= cap` that are the same for every n in the interval,
    /// or `None` if two of them coincide.
    ///
    /// n ↦ n^j is increasing, so those digits are exactly the common top
    /// digits of `a^j` and `e^j`, scanned down to the first disagreement.
    /// The scan starts at `ndig(e^j − a^j)`: the two cannot agree on every
    /// position `>= i` unless `e^j − a^j < b^i`.
    #[must_use]
    pub fn cert(&self, a: u128, e: u128, cap: u32) -> Option<u64> {
        let a2 = W4::mul_u128_u128(a, a);
        let e2 = W4::mul_u128_u128(e, e);
        let a3 = a2.mul_u128(a);
        let e3 = e2.mul_u128(e);
        let mut mask = 0u64;
        let mut dx = [0u8; DIGIT_BUF];
        let mut dy = [0u8; DIGIT_BUF];
        for (x, y, sp) in [(a2, e2, self.s2), (a3, e3, self.s3)] {
            let c0 = cap.max(self.ndig_w(&y.sub(&x)));
            if c0 >= sp {
                continue;
            }
            let len = sp - c0;
            let qx = self.shift_down(x, c0);
            let qy = self.shift_down(y, c0);
            self.digits_w(qx, len, &mut dx);
            self.digits_w(qy, len, &mut dy);
            for i in (0..len as usize).rev() {
                if dx[i] != dy[i] {
                    break;
                }
                let bit = 1u64 << dx[i];
                if mask & bit != 0 {
                    return None;
                }
                mask |= bit;
            }
        }
        Some(mask)
    }

    /// Lowest output position the certificate of `[a, e]` (cap `cap`) can
    /// cover, over both powers: every certified digit of n² (n³) sits at a
    /// position `>= max(cap, ndig(e² − a²))` (resp. the cube).
    #[must_use]
    pub fn cert_floor(&self, a: u128, e: u128, cap: u32) -> u32 {
        let a2 = W4::mul_u128_u128(a, a);
        let e2 = W4::mul_u128_u128(e, e);
        let a3 = a2.mul_u128(a);
        let e3 = e2.mul_u128(e);
        let c2 = cap.max(self.ndig_w(&e2.sub(&a2)));
        let c3 = cap.max(self.ndig_w(&e3.sub(&a3)));
        c2.min(c3)
    }

    /// Top prefixes of depth `depth` meeting `[s, e_incl]` whose certificate
    /// (cap `cap`) passes, breadth first; a failing prefix's children all
    /// fail (a sub-interval's common digits include its parent's).
    #[must_use]
    #[allow(clippy::many_single_char_names)] // b, w, p, a, e as in the docs
    pub fn top_layer(&self, s: u128, e_incl: u128, depth: u32, cap: u32) -> Vec<(u128, u64)> {
        let b = u128::from(self.b);
        let mut level: Vec<(u128, u64)> = vec![(0, 0)];
        for j in 1..=depth {
            let w = self.pow[(self.l - j) as usize];
            let (plo, phi) = (s / w, e_incl / w);
            let mut next = Vec::new();
            for &(par, _) in &level {
                let first = (par * b).max(plo);
                let last = (par * b + b - 1).min(phi);
                let mut p = first;
                while p <= last {
                    let a = (p * w).max(s);
                    let e = (p * w + w - 1).min(e_incl);
                    if let Some(m) = self.cert(a, e, cap) {
                        next.push((p, m));
                    }
                    p += 1;
                }
            }
            level = next;
        }
        level
    }

    /// Bottom residues extended from `r` (depth `j`, certificate `mask`) to
    /// depth `k`, with positions `[f0, f0 + pp)` forced to the digits of `v`.
    /// Digit `j` of `(r + d·b^j)²` is `(⌊r²/b^j⌋ + 2d·(r mod b)) mod b` and of
    /// the cube `(⌊r³/b^j⌋ + 3d·(r mod b)²) mod b` (`d²`, `d³` at `j = 0`).
    // The recursion's state is the argument list; a residue is below
    // b^k < 2^40 (`JoinParams::supported`), so it fits u64.
    #[allow(
        clippy::too_many_arguments,
        clippy::cast_possible_truncation,
        clippy::many_single_char_names
    )]
    pub fn bot_dfs(
        &self,
        r: u128,
        mask: u64,
        j: u32,
        k: u32,
        f0: u32,
        pp: u32,
        v: u128,
        out: &mut Vec<(u64, u64)>,
    ) {
        if j == k {
            out.push((r as u64, mask));
            return;
        }
        let b = u128::from(self.b);
        let (dlo, dhi) = if j >= f0 && j < f0 + pp {
            let d = (v / self.pow[(j - f0) as usize]) % b;
            (d, d)
        } else {
            (0, b - 1)
        };
        let (mut u2, mut u3, mut s2, mut s3) = (0u128, 0u128, 0u128, 0u128);
        if j > 0 {
            let r2 = r * r;
            let r3 = r2 * r;
            u2 = (r2 / self.pow[j as usize]) % b;
            u3 = (r3 / self.pow[j as usize]) % b;
            let r0 = r % b;
            s2 = 2 * r0 % b;
            s3 = 3 * (r0 * r0 % b) % b;
        }
        let pj = self.pow[j as usize];
        for d in dlo..=dhi {
            let (g2, g3) = if j == 0 {
                (d * d % b, d * d % b * d % b)
            } else {
                ((u2 + d * s2) % b, (u3 + d * s3) % b)
            };
            if g2 == g3 {
                continue;
            }
            let (m2, m3) = (1u64 << g2, 1u64 << g3);
            if mask & (m2 | m3) != 0 {
                continue;
            }
            self.bot_dfs(r + d * pj, mask | m2 | m3, j + 1, k, f0, pp, v, out);
        }
    }
}

/// The host side of a field: everything that does not depend on the
/// partition value.
#[derive(Clone)]
pub struct FieldSetup {
    pub base: Base,
    pub b: u32,
    pub s: u128,
    pub e: u128,
    pub jp: JoinParams,
    pub f0: u32,
    pub key_level: bool,
    /// `b^f0`: the width of one top prefix block.
    pub w: u128,
    /// `b^p`: the number of partition values.
    pub nparts: u128,
    pub m1: u32,
    /// Buckets per partition: key values × digit-sum classes.
    pub nb: usize,
    pub plo: u128,
    pub phi: u128,
    /// Top layer at depth `t − p`.
    pub tlay: Vec<(u128, u64)>,
    /// `bpre`: residues mod `b^f0` whose `2·f0` low output digits are
    /// distinct, sorted by digit-sum class mod `b − 1`; `seg[c]..seg[c + 1]`
    /// is class `c`.
    pub bp_r: Vec<u32>,
    pub bp_m: Vec<u64>,
    pub seg: Vec<u32>,
    /// The prefilter tests the output digits below `k2` (1 or 2 more than
    /// the bottom list's `k`).
    pub k2: u32,
    /// Certificate floor of every full-width block in the field (monotone in
    /// `P`, so the first full block bounds them all).
    pub full_floor: u32,
    pub secs: f64,
}

impl FieldSetup {
    /// Set up `[s, e)` at base `b` with parameters `jp`.
    ///
    /// # Errors
    /// A field or parameters the join cannot take: an empty field, one that
    /// crosses a digit-length boundary, parameters `jp` does not support at
    /// this length, `n` of 2^96 or more, or no room for the prefilter.
    #[allow(clippy::many_single_char_names)] // b, s, e, t, k, o, w as in `overlap_join`
    pub fn new(b: u32, s: u128, e: u128, jp: JoinParams) -> Result<Self> {
        let t0 = Instant::now();
        ensure!(s < e, "empty field");
        let base = Base::try_new(b, s, e - 1)
            .ok_or_else(|| anyhow!("[{s}, {e}) crosses a digit-length boundary in base {b}"))?;
        let l = base.l;
        ensure!(
            jp.supported(b, l),
            "{jp:?} not supported at base {b}, L = {l}"
        );
        ensure!(e - 1 < 1 << 96, "device tops need n < 2^96");
        let (t, k, pp) = (jp.t, jp.k, jp.p);
        let f0 = l - t;
        let o = t + k - l;
        let w = base.powu(f0);
        let (tlay, full_floor) = Self::top_side(&base, jp, w, s, e);
        let mut bpre: Vec<(u64, u64)> = Vec::new();
        base.bot_dfs(0, 0, 0, f0, f0, 0, 0, &mut bpre);
        ensure!(!bpre.is_empty(), "empty bottom list");
        let m1 = b - 1;
        bpre.sort_unstable_by_key(|&(r, _)| (r % u64::from(m1), r));
        let mut seg = vec![0u32; m1 as usize + 1];
        for &(r, _) in &bpre {
            seg[usize::try_from(r % u64::from(m1))? + 1] += 1;
        }
        for c in 0..m1 as usize {
            seg[c + 1] += seg[c];
        }
        // Residues below b^f0 < 2^32 (`JoinParams::supported`).
        let bp_r = bpre
            .iter()
            .map(|&(r, _)| u32::try_from(r))
            .collect::<Result<_, _>>()?;
        let bp_m = bpre.iter().map(|&(_, m)| m).collect();
        let keyspace = base.powu(o - pp);
        // The prefilter tests digits k..k2 too; it needs both powers to have
        // at least k2 digits and P mod b^(k2 - f0) to fit u32.
        let mut mid = 2;
        while mid > 0
            && (k + mid > base.s2 || u64::from(b).pow(k + mid - f0) >= 1 << 32 || k + mid > 12)
        {
            mid -= 1;
        }
        // The device stage always prefilters (every field the gate takes, and
        // every test window, has a depth of 2).
        ensure!(
            mid > 0,
            "{jp:?} at base {b} leaves no room for the prefilter"
        );
        Ok(Self {
            k2: k + mid,
            full_floor,
            b,
            s,
            e,
            jp,
            f0,
            key_level: o - pp == 1,
            w,
            nparts: base.powu(pp),
            m1,
            nb: keyspace as usize * m1 as usize,
            plo: s / w,
            phi: (e - 1) / w,
            tlay,
            bp_r,
            bp_m,
            seg,
            secs: t0.elapsed().as_secs_f64(),
            base,
        })
    }

    /// The field's top side over `[s, e)`: its top layer at depth `t − p`,
    /// and the certificate floor of its full-width blocks.
    fn top_side(base: &Base, jp: JoinParams, w: u128, s: u128, e: u128) -> (Vec<(u128, u64)>, u32) {
        let tlay = base.top_layer(s, e - 1, jp.t - jp.p, jp.k);
        let first_full = s.div_ceil(w);
        let full_floor = if first_full * w + w - 1 < e {
            base.cert_floor(first_full * w, first_full * w + w - 1, jp.k)
        } else {
            0
        };
        (tlay, full_floor)
    }

    /// The same setup for `[s, e)`, another range with the same digit
    /// lengths (a slice of the field, see [`join_slices`]): the bottom side
    /// is reused, only the top side is built again.
    ///
    /// # Errors
    /// An empty range, or one whose digit lengths differ from this field's.
    pub fn sub_range(&self, s: u128, e: u128) -> Result<Self> {
        let t0 = Instant::now();
        ensure!(s < e, "empty field");
        let lengths = Base::try_new(self.b, s, e - 1).map(|b| (b.l, b.s2, b.s3));
        ensure!(
            lengths == Some((self.base.l, self.base.s2, self.base.s3)),
            "[{s}, {e}) does not share the digit lengths of [{}, {})",
            self.s,
            self.e
        );
        let (tlay, full_floor) = Self::top_side(&self.base, self.jp, self.w, s, e);
        Ok(Self {
            base: self.base.clone(),
            b: self.b,
            s,
            e,
            jp: self.jp,
            f0: self.f0,
            key_level: self.key_level,
            w: self.w,
            nparts: self.nparts,
            m1: self.m1,
            nb: self.nb,
            plo: s / self.w,
            phi: (e - 1) / self.w,
            tlay,
            bp_r: self.bp_r.clone(),
            bp_m: self.bp_m.clone(),
            seg: self.seg.clone(),
            k2: self.k2,
            full_floor,
            secs: t0.elapsed().as_secs_f64(),
        })
    }
}

/// What the reference join found on a set of partitions.
#[derive(Default, Debug, Clone)]
pub struct JoinStats {
    /// Key-equal (top, bottom) pairs inside the field.
    pub matches: u64,
    /// Pairs that passed the AND (fully checked).
    pub survivors: u64,
    pub hits: Vec<u128>,
}

/// The reference join over `[s, e)` on the partition values `parts` (`None`
/// = all `b^p`): every survivor is fully checked with `get_is_nice`, and
/// recorded in `record` if given. The GPU stage must produce exactly these
/// survivors.
///
/// # Panics
/// If `jp` does not fit the field's digit length (`t + k > L`, `t <= L`,
/// `k < L`, `p <= t + k - L`).
// The arithmetic is in the module docs' names (t, k, p, o, ...); residues
// below b^k < 2^40 and bucket indices fit their narrower types. Some callers
// want only `record`.
#[allow(
    clippy::many_single_char_names,
    clippy::cast_possible_truncation,
    clippy::too_many_lines,
    clippy::must_use_candidate
)]
pub fn join_range(
    base: &Base,
    s: u128,
    e: u128,
    jp: JoinParams,
    parts: Option<&[u128]>,
    mut record: Option<&mut Vec<u128>>,
) -> JoinStats {
    let mut st = JoinStats::default();
    let l = base.l;
    let (t, k, pp) = (jp.t, jp.k, jp.p);
    assert!(
        t + k > l && t <= l && k < l,
        "need t+k>L, t<=L, k<L (L={l})"
    );
    let f0 = l - t;
    let o = t + k - l;
    assert!(pp <= o);
    let e_incl = e - 1;
    let tlay = base.top_layer(s, e_incl, t - pp, k);
    let mut bpre = Vec::new();
    base.bot_dfs(0, 0, 0, f0, f0, 0, 0, &mut bpre);
    let nv = base.pow[pp as usize];
    let all: Vec<u128>;
    let parts = if let Some(p) = parts {
        p
    } else {
        all = (0..nv).collect();
        &all
    };
    let keyspace = base.pow[(o - pp) as usize] as usize;
    let m1 = base.m1 as usize;
    let nb = keyspace * m1;
    let kdiv = base.pow[(f0 + pp) as usize] as u64;
    let lowmod = base.pow[f0 as usize] as u64;
    let pdiv = base.pow[pp as usize];
    let w_t = base.pow[f0 as usize];
    let (plo, phi) = (s / w_t, e_incl / w_t);
    let bkt = |r: u64| ((r / kdiv) as usize) * m1 + ((r % lowmod) % m1 as u64) as usize;
    for &v in parts {
        let tops: Vec<(u128, u64)> = tlay
            .iter()
            .filter_map(|&(p0, _)| {
                let p = p0 * pdiv + v;
                if p < plo || p > phi {
                    return None;
                }
                let a = (p * w_t).max(s);
                let ee = (p * w_t + w_t - 1).min(e_incl);
                base.cert(a, ee, k).map(|m| (p, m))
            })
            .collect();
        if tops.is_empty() {
            continue;
        }
        let mut bl = Vec::new();
        for &(r, m) in &bpre {
            base.bot_dfs(u128::from(r), m, f0, k, f0, pp, v, &mut bl);
        }
        let mut off = vec![0u32; nb + 1];
        for &(r, _) in &bl {
            off[bkt(r) + 1] += 1;
        }
        for x in 0..nb {
            off[x + 1] += off[x];
        }
        let mut cur = off.clone();
        let mut sorted = vec![(0u64, 0u64); bl.len()];
        for &(r, m) in &bl {
            let kx = bkt(r);
            sorted[cur[kx] as usize] = (r % lowmod, m);
            cur[kx] += 1;
        }
        for &(p, tmask) in &tops {
            let key = ((p / pdiv) % keyspace as u128) as usize;
            let pc = (p % m1 as u128) as usize;
            for &root in &base.roots {
                let bx = key * m1 + (root as usize + m1 - pc) % m1;
                for &(rl, bm) in &sorted[off[bx] as usize..off[bx + 1] as usize] {
                    let n = p * w_t + u128::from(rl);
                    if n < s || n >= e {
                        continue;
                    }
                    st.matches += 1;
                    if tmask & bm != 0 {
                        continue;
                    }
                    st.survivors += 1;
                    if let Some(rec) = record.as_deref_mut() {
                        rec.push(n);
                    }
                    if get_is_nice(n, base.b) {
                        st.hits.push(n);
                    }
                }
            }
        }
    }
    st
}

/// Fields the tests share: production-size fields that the join takes.
#[cfg(test)]
pub(crate) mod test_fields {
    use super::FieldSetup;

    /// A frontier field of base 57.
    pub(crate) const FRONTIER_57: u128 = 28_151_599_893_042_801_193;

    /// Production-size fields at bases 42-64 (gate-sized at 42, 1e14
    /// elsewhere), from the middle of their bands and the frontier.
    pub(crate) const PLAN_FIELDS: &[(u32, u128, u128)] = &[
        (42, 9_682_651_996_416, 10_000_000_000_000),
        (50, 62_082_117_268_529_817, 100_000_000_000_000),
        (57, FRONTIER_57, 100_000_000_000_000),
        (60, 1_366_405_974_057_412_100_454, 100_000_000_000_000),
        (62, 7_997_740_455_941_656_911_841, 100_000_000_000_000),
        (64, 41_242_006_262_957_161_709_568, 100_000_000_000_000),
    ];

    /// Dan Stoyell's exactness windows: several shapes, aligned and
    /// unaligned, some where the client's MSD filter lets candidates through.
    pub(crate) const WINDOWS: &[(u32, u128, u128, u32, u32, u32)] = &[
        (20, 58_945, 160_000, 3, 2, 0),
        (20, 58_945, 160_000, 2, 3, 1),
        (20, 60_001, 150_003, 2, 3, 1),
        (25, 3_339_797, 4_339_797, 4, 3, 1),
        (25, 5_000_123, 5_700_456, 3, 4, 2),
        (30, 300_000_000, 301_000_000, 5, 4, 2),
        (34, 12_000_000_017, 12_001_000_017, 5, 5, 2),
        (40, 3_000_000_000_000, 3_000_002_000_000, 6, 5, 2),
        (40, 3_000_000_000_000, 3_000_002_000_000, 7, 4, 2),
        (
            57,
            30_000_000_000_000_000_000,
            30_000_000_000_000_300_000,
            10,
            4,
            1,
        ),
        (
            57,
            20_635_899_893_042_801_193,
            20_635_899_893_043_101_193,
            9,
            5,
            1,
        ),
        (
            57,
            78_920_310_198_429_586_458,
            78_920_310_198_429_886_458,
            9,
            5,
            2,
        ),
        (
            58,
            114_041_927_169_846_138_720,
            114_041_927_169_846_438_720,
            10,
            4,
            2,
        ),
        (
            60,
            1_573_714_731_429_953_349_518,
            1_573_714_731_429_953_649_518,
            10,
            4,
            2,
        ),
        (
            64,
            52_125_117_810_081_128_433_988,
            52_125_117_810_081_128_733_988,
            11,
            4,
            2,
        ),
    ];

    /// Partition values that hold at least one top prefix of the field.
    pub(crate) fn live_partitions(fs: &FieldSetup) -> Vec<u32> {
        let mut v: Vec<u32> = (fs.plo..=fs.phi)
            .map(|p| u32::try_from(p % fs.nparts).expect("b^p < 2^32"))
            .take(usize::try_from(fs.nparts).expect("b^p < 2^32") + 1)
            .collect();
        v.sort_unstable();
        v.dedup();
        v
    }

    /// The prefilter by its definition: the digits at positions `0..k2` of
    /// n² and n³ (mod `b^k2` in u128) are distinct, and disjoint from the
    /// top certificate when every certified position is `>= k2`.
    #[allow(clippy::many_single_char_names)] // b, n, r, p, a as in the docs
    pub(crate) fn mid_mirror(fs: &FieldSetup, n: u128) -> bool {
        let b = u128::from(fs.b);
        let bk = b.pow(fs.k2);
        let r = n % bk;
        let sq = r * r % bk;
        let cu = sq * r % bk;
        let p = n / fs.w;
        let a = (p * fs.w).max(fs.s);
        let ee = (p * fs.w + fs.w - 1).min(fs.e - 1);
        let mut seen = 0u64;
        let floor = if a == p * fs.w && ee == p * fs.w + fs.w - 1 {
            fs.full_floor
        } else {
            fs.base.cert_floor(a, ee, fs.jp.k)
        };
        if floor >= fs.k2 {
            seen = fs
                .base
                .cert(a, ee, fs.jp.k)
                .expect("a survivor's top is certified");
        }
        let (mut x2, mut x3) = (sq, cu);
        for _ in 0..fs.k2 {
            for d in [x2 % b, x3 % b] {
                let bit = 1u64 << d;
                if seen & bit != 0 {
                    return false;
                }
                seen |= bit;
            }
            x2 /= b;
            x3 /= b;
        }
        true
    }

    /// The reference join over every partition of a field, on every core:
    /// its join survivors, prefilter survivors (by [`mid_mirror`]) and hits,
    /// sorted. Minutes of CPU for a production-size field.
    pub(crate) fn reference_field(fs: &FieldSetup) -> (u64, u64, Vec<u128>) {
        let threads = std::thread::available_parallelism().map_or(4, usize::from);
        let parts: Vec<u128> = (0..fs.nparts).collect();
        let (survivors, checked, mut hits) = std::thread::scope(|sc| {
            let workers: Vec<_> = parts
                .chunks(parts.len().div_ceil(threads))
                .map(|chunk| {
                    sc.spawn(move || {
                        // One partition at a time: a whole field's survivors
                        // (4.5e8 at base 42) would not fit in memory.
                        let (mut survivors, mut checked, mut hits) = (0u64, 0u64, Vec::new());
                        let mut rec = Vec::new();
                        for &v in chunk {
                            rec.clear();
                            let st = super::join_range(
                                &fs.base,
                                fs.s,
                                fs.e,
                                fs.jp,
                                Some(&[v]),
                                Some(&mut rec),
                            );
                            survivors += st.survivors;
                            hits.extend(st.hits);
                            checked += rec.iter().filter(|&&n| mid_mirror(fs, n)).count() as u64;
                        }
                        (survivors, checked, hits)
                    })
                })
                .collect();
            workers
                .into_iter()
                .map(|w| w.join().expect("reference worker"))
                .fold((0u64, 0u64, Vec::new()), |(a, b, mut h), (x, y, z)| {
                    h.extend(z);
                    (a + x, b + y, h)
                })
        });
        hits.sort_unstable();
        (survivors, checked, hits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base_range::get_base_range_u128;

    /// Base-b digits of x, least significant first.
    fn digits(mut x: u128, b: u128, n: usize) -> Vec<u128> {
        let mut v = Vec::with_capacity(n);
        for _ in 0..n {
            v.push(x % b);
            x /= b;
        }
        v
    }

    /// A splitmix64 stream for the property tests.
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }

        /// Below `n` (> 0), near enough uniform for a test.
        fn below(&mut self, n: u128) -> u128 {
            ((u128::from(self.next()) << 64) | u128::from(self.next())) % n
        }
    }

    /// The digit values of n² and n³ at positions `>= cap`, and whether one
    /// of them repeats.
    fn top_digits(base: &Base, n: u128, cap: u32) -> (u64, bool) {
        let b = u128::from(base.b);
        let (mut seen, mut repeat) = (0u64, false);
        for (x, len) in [(n * n, base.s2), (n * n * n, base.s3)] {
            for d in digits(x, b, len as usize).into_iter().skip(cap as usize) {
                let bit = 1u64 << d;
                repeat |= seen & bit != 0;
                seen |= bit;
            }
        }
        (seen, repeat)
    }

    /// `cert` against its definition, for every n of random intervals: a
    /// mask only if every n has all of its digits among those of n² and n³
    /// at positions `>= cap` (so the join's rejection of a bottom digit in
    /// it is a real repeat), and `None` only if every n repeats a digit
    /// there. Raising the cap to `cert_floor` changes nothing, which is what
    /// lets the prefilter treat the certificate as sitting at or above it.
    #[test]
    fn certificates_hold_for_every_n_of_their_interval() {
        let mut rng = Rng(0x0BAD_5EED);
        let (mut masks, mut collisions) = (0, 0);
        for b in [5u32, 7, 10, 12, 16, 20, 23, 31, 40] {
            let Ok(Some(r)) = get_base_range_u128(b) else {
                continue;
            };
            let span = r.range_end - r.range_start;
            for _ in 0..60 {
                let len = 1 + rng.below(span.min(400));
                let a = r.range_start + rng.below(span - len + 1);
                let e = a + len - 1;
                let Some(base) = Base::try_new(b, a, e) else {
                    continue; // n² or n³ changes length inside the interval
                };
                let cap = u32::try_from(rng.below(u128::from(base.s3) + 1)).unwrap();
                let cert = base.cert(a, e, cap);
                let floor = base.cert_floor(a, e, cap);
                assert_eq!(base.cert(a, e, floor), cert, "b{b} [{a}, {e}] cap {cap}");
                match cert {
                    Some(_) => masks += 1,
                    None => collisions += 1,
                }
                for n in a..=e {
                    let (seen, repeat) = top_digits(&base, n, cap);
                    match cert {
                        Some(m) => assert_eq!(
                            m & !seen,
                            0,
                            "b{b} [{a}, {e}] cap {cap}: {n} lacks a certified digit"
                        ),
                        None => assert!(
                            repeat,
                            "b{b} [{a}, {e}] cap {cap}: {n} has no repeat at those positions"
                        ),
                    }
                }
            }
        }
        assert!(
            masks > 50 && collisions > 50,
            "{masks} certificates, {collisions} collisions: a case is barely tested"
        );
    }

    /// `bot_dfs` against brute force: exactly the residues r mod b^k whose
    /// digits `f0..f0 + pp` are v's and whose k low digits of r² and of r³
    /// are 2k distinct values, each with that set as its mask.
    #[test]
    fn bottom_residues_match_their_definition() {
        for (b, k, f0, pp) in [
            (7u32, 3u32, 1u32, 1u32),
            (10, 3, 0, 2),
            (12, 3, 1, 1),
            (16, 3, 2, 1),
            (20, 3, 1, 2),
            (31, 2, 0, 1),
        ] {
            // The bottom list depends on the base alone, not on a field.
            let base = Base::try_new(b, u128::from(b), u128::from(b)).unwrap();
            let bb = u128::from(b);
            let bk = bb.pow(k);
            let (wf, wp) = (bb.pow(f0), bb.pow(pp));
            for v in [0, 1, wp / 2, wp - 1] {
                let mut got = Vec::new();
                base.bot_dfs(0, 0, 0, k, f0, pp, v, &mut got);
                got.sort_unstable();
                let mut want = Vec::new();
                for x in (0..bk).filter(|x| (x / wf) % wp == v) {
                    let low = digits(x * x % bk, bb, k as usize).into_iter().chain(digits(
                        x * x * x % bk,
                        bb,
                        k as usize,
                    ));
                    let (mut mask, mut distinct) = (0u64, true);
                    for d in low {
                        distinct &= mask & (1 << d) == 0;
                        mask |= 1 << d;
                    }
                    if distinct {
                        want.push((u64::try_from(x).unwrap(), mask));
                    }
                }
                assert_eq!(got, want, "b{b} k {k} f0 {f0} pp {pp} v {v}");
            }
        }
    }

    /// The top layer keeps the prefix of every n whose digits of n² and n³
    /// at positions `>= cap` are distinct, with a mask that n really has.
    #[test]
    #[allow(clippy::many_single_char_names)]
    fn top_layer_keeps_every_viable_prefix() {
        let mut rng = Rng(0x7095);
        let mut viable = 0u64;
        for b in [10u32, 12, 16, 20, 23] {
            let Ok(Some(r)) = get_base_range_u128(b) else {
                continue;
            };
            let span = r.range_end - r.range_start;
            for _ in 0..4 {
                let len = span.min(5_000);
                let s = r.range_start + rng.below(span - len + 1);
                let e = s + len - 1;
                let Some(base) = Base::try_new(b, s, e) else {
                    continue;
                };
                for depth in 1..base.l.min(5) {
                    let cap = u32::try_from(rng.below(u128::from(base.s2))).unwrap();
                    let layer: std::collections::HashMap<u128, u64> =
                        base.top_layer(s, e, depth, cap).into_iter().collect();
                    let w = u128::from(b).pow(base.l - depth);
                    for n in s..=e {
                        let (seen, repeat) = top_digits(&base, n, cap);
                        if repeat {
                            continue;
                        }
                        viable += 1;
                        let m = layer.get(&(n / w)).unwrap_or_else(|| {
                            panic!("b{b} depth {depth} cap {cap}: the prefix of {n} was dropped")
                        });
                        assert_eq!(m & !seen, 0, "b{b} depth {depth} cap {cap}: {n}");
                    }
                }
            }
        }
        assert!(
            viable > 1_000,
            "only {viable} viable n: the test is not testing"
        );
    }

    /// The survivors of the same certificates, n by n: the residue class,
    /// the k low digits of n² and n³, the top certificate of n's prefix
    /// block, and the AND. Returns (pairs matched, survivors).
    fn brute(base: &Base, s: u128, e: u128, jp: JoinParams) -> (u64, Vec<u128>) {
        let bb = u128::from(base.b);
        let w_t = bb.pow(base.l - jp.t);
        let bk = bb.pow(jp.k);
        let roots: Vec<u128> = base.roots.iter().map(|&r| u128::from(r)).collect();
        let (mut matches, mut surv) = (0u64, Vec::new());
        for n in s..e {
            if !roots.contains(&(n % (bb - 1))) {
                continue;
            }
            let r = n % bk;
            let d2 = digits((r * r) % bk, bb, jp.k as usize);
            let d3 = digits(((r * r) % bk * r) % bk, bb, jp.k as usize);
            let mut bm = 0u64;
            let mut ok = true;
            for &d in d2.iter().chain(d3.iter()) {
                let bit = 1u64 << d;
                if bm & bit != 0 {
                    ok = false;
                    break;
                }
                bm |= bit;
            }
            if !ok {
                continue;
            }
            let p = n / w_t;
            let lo = (p * w_t).max(s);
            let hi = (p * w_t + w_t - 1).min(e - 1);
            let Some(tm) = base.cert(lo, hi, jp.k) else {
                continue;
            };
            matches += 1;
            if tm & bm == 0 {
                surv.push(n);
            }
        }
        (matches, surv)
    }

    /// Dan Stoyell's exactness windows (aligned and unaligned, several
    /// shapes, some where the client's MSD filter lets candidates through),
    /// cut to the ones that run in a second or two in a debug build.
    const WINDOWS: &[(u32, u128, u128, u32, u32, u32)] = &[
        (20, 58_945, 160_000, 3, 2, 0),
        (20, 58_945, 160_000, 3, 2, 1),
        (20, 58_945, 160_000, 2, 3, 1),
        (20, 60_001, 150_003, 2, 3, 1),
        (25, 3_339_797, 3_539_797, 4, 3, 1),
        (25, 5_000_123, 5_200_456, 3, 4, 2),
        (30, 300_000_000, 300_200_000, 5, 4, 2),
        (34, 12_000_000_017, 12_000_200_017, 5, 5, 2),
        (40, 3_000_000_000_000, 3_000_000_300_000, 6, 5, 2),
        (40, 3_000_000_000_000, 3_000_000_300_000, 7, 4, 2),
        (
            57,
            30_000_000_000_000_000_000,
            30_000_000_000_000_100_000,
            10,
            4,
            1,
        ),
        (
            57,
            20_635_899_893_042_801_193,
            20_635_899_893_042_901_193,
            9,
            5,
            1,
        ),
        (
            57,
            78_920_310_198_429_586_458,
            78_920_310_198_429_686_458,
            9,
            5,
            2,
        ),
        (
            60,
            1_573_714_731_429_953_349_518,
            1_573_714_731_429_953_449_518,
            10,
            4,
            2,
        ),
        (
            64,
            52_125_117_810_081_128_433_988,
            52_125_117_810_081_128_533_988,
            11,
            4,
            2,
        ),
    ];

    #[test]
    fn reference_join_equals_brute_force() {
        for &(b, s, e, t, k, p) in WINDOWS {
            let base = Base::try_new(b, s, e - 1).expect("window inside one length class");
            let jp = JoinParams { t, k, p };
            let mut rec = Vec::new();
            let st = join_range(&base, s, e, jp, None, Some(&mut rec));
            rec.sort_unstable();
            let (bm, bs) = brute(&base, s, e, jp);
            assert_eq!(rec, bs, "b{b} [{s}, {e}) {jp:?}: survivors differ");
            assert_eq!(st.matches, bm, "b{b} [{s}, {e}) {jp:?}: matches differ");
            assert_eq!(st.survivors, u64::try_from(rec.len()).unwrap());
        }
    }

    #[test]
    fn reference_join_finds_69_in_base_10() {
        let base = Base::try_new(10, 47, 99).unwrap();
        let st = join_range(&base, 47, 100, JoinParams { t: 2, k: 1, p: 0 }, None, None);
        assert_eq!(st.hits, vec![69]);
    }

    /// The join must keep every n whose digits are distinct on the positions
    /// it certifies, which a nice n always is. Plant the property directly:
    /// at base 12 the band holds no nice number, but every survivor of the
    /// brute force must also be a survivor of the join for every shape, and
    /// the join must not keep anything else.
    #[test]
    fn reference_join_matches_brute_force_on_a_whole_small_band() {
        let r = get_base_range_u128(12).unwrap().unwrap();
        let (s, e) = (r.range_start, r.range_end);
        for jp in [
            JoinParams { t: 2, k: 2, p: 0 },
            JoinParams { t: 2, k: 2, p: 1 },
            JoinParams { t: 3, k: 2, p: 1 },
        ] {
            let base = Base::try_new(12, s, e - 1).unwrap();
            let mut rec = Vec::new();
            join_range(&base, s, e, jp, None, Some(&mut rec));
            rec.sort_unstable();
            assert_eq!(rec, brute(&base, s, e, jp).1, "base 12 {jp:?}");
        }
    }

    #[test]
    fn production_parameters_are_supported_across_the_join_bases() {
        for base in JOIN_MIN_BASE..=JOIN_MAX_BASE {
            let Ok(Some(r)) = get_base_range_u128(base) else {
                continue;
            };
            if crate::residue_filter::get_residue_filter_u128(&base).is_empty() {
                continue;
            }
            if r.range_end - r.range_start < JOIN_MIN_FIELD_SIZE {
                continue;
            }
            // The first and last full-size fields of the band.
            for start in [r.range_start, r.range_end - JOIN_MIN_FIELD_SIZE] {
                let f = FieldSize::new(start, start + JOIN_MIN_FIELD_SIZE);
                let jp = join_params_for(base, &f)
                    .unwrap_or_else(|| panic!("base {base}: no join parameters for {f:?}"));
                let base_w = Base::try_new(base, f.first(), f.last()).unwrap();
                assert!(jp.supported(base, base_w.l), "base {base}: {jp:?}");
            }
        }
    }

    #[test]
    fn small_fields_and_small_bases_stay_on_the_stride_pipeline() {
        let r = get_base_range_u128(57).unwrap().unwrap();
        let s = r.range_start;
        assert!(join_params_for(57, &FieldSize::new(s, s + 4_000_000_000)).is_none());
        assert!(join_params_for(57, &FieldSize::new(s, s + JOIN_MIN_FIELD_SIZE)).is_some());
        let r = get_base_range_u128(35).unwrap().unwrap();
        let end = (r.range_start + JOIN_MIN_FIELD_SIZE).min(r.range_end);
        assert!(join_params_for(35, &FieldSize::new(r.range_start, end)).is_none());
    }

    /// Slices cover the field in order, meet at prefix-block boundaries, hold
    /// at most the cap's prefixes, and are as few as the cap allows.
    #[test]
    fn slices_cover_the_field_at_prefix_boundaries() {
        let block = 1_000u128;
        for (s, e, cap) in [
            (0u128, 10_000u128, 3u128),
            (1_234, 9_876, 2),
            (5_000, 5_001, 7),
            (999, 1_001, 1),
            (0, 10_000, 100),
        ] {
            let slices = join_slices(&FieldSize::new(s, e), block, cap);
            assert_eq!(slices.first().map(FieldSize::start), Some(s));
            assert_eq!(slices.last().map(FieldSize::end), Some(e));
            for w in slices.windows(2) {
                assert_eq!(w[0].end(), w[1].start());
                assert_eq!(w[1].start() % block, 0);
            }
            for sl in &slices {
                assert!((sl.end() - 1) / block - sl.start() / block < cap, "{sl:?}");
            }
            let prefixes = (e - 1) / block - s / block + 1;
            assert_eq!(
                slices.len() as u128,
                prefixes.div_ceil(cap),
                "[{s}, {e}) cap {cap}"
            );
        }
    }

    /// A slice set up from the field's setup is the setup of that range as a
    /// field of its own, and the slices' top layers make up the field's.
    #[test]
    fn a_slice_sets_up_as_a_field_of_its_own() {
        let (b, s) = (42, 9_682_651_996_416u128);
        let range = FieldSize::new(s, s + JOIN_MIN_FIELD_SIZE);
        let jp = join_params_for(b, &range).expect("a join field");
        let whole = FieldSetup::new(b, range.start(), range.end(), jp).expect("setup");
        let block = prefix_block(b, whole.base.l, jp);
        let prefixes = (range.last() / block - range.start() / block + 1) / 3 + 1;
        let slices = join_slices(&range, block, prefixes);
        assert_eq!(slices.len(), 3);
        let mut layer = Vec::new();
        for sl in &slices {
            let sub = whole.sub_range(sl.start(), sl.end()).expect("slice setup");
            let own = FieldSetup::new(b, sl.start(), sl.end(), jp).expect("setup");
            assert_eq!(
                (sub.s, sub.e, sub.plo, sub.phi, sub.full_floor, sub.k2),
                (own.s, own.e, own.plo, own.phi, own.full_floor, own.k2)
            );
            assert_eq!(sub.tlay, own.tlay);
            assert_eq!(
                (&sub.bp_r, &sub.bp_m, &sub.seg),
                (&own.bp_r, &own.bp_m, &own.seg)
            );
            layer.extend(sub.tlay);
        }
        assert_eq!(layer, whole.tlay);
        // Another digit length is refused.
        assert!(whole.sub_range(1, 2).is_err());
    }

    #[test]
    fn the_verdict_names_why_a_field_takes_the_stride_path() {
        // As routed without `NICE_JOIN_ROUTE`, whatever the environment.
        let verdict = |b, f: &FieldSize| join_verdict_routed(b, f, RouteOverride::Auto);
        let r = get_base_range_u128(57).unwrap().unwrap();
        let s = r.range_start;
        assert_eq!(
            verdict(57, &FieldSize::new(s, s + 4_000_000_000)),
            Err(StrideReason::BelowMinSize)
        );
        assert!(verdict(57, &FieldSize::new(s, s + JOIN_MIN_FIELD_SIZE)).is_ok());
        let r = get_base_range_u128(35).unwrap().unwrap();
        let end = (r.range_start + JOIN_MIN_FIELD_SIZE).min(r.range_end);
        assert_eq!(
            verdict(35, &FieldSize::new(r.range_start, end)),
            Err(StrideReason::Base)
        );
        // Across 50^10, where n gains a digit.
        let edge = 50u128.pow(10);
        assert_eq!(
            verdict(
                50,
                &FieldSize::new(edge - 5_000_000_000_000, edge + 5_000_000_000_000)
            ),
            Err(StrideReason::LengthChange)
        );
    }

    /// `NICE_JOIN_ROUTE`: `stride` keeps every field off the join, `join`
    /// lifts only the size cutoff (base, length and width still decide), and
    /// anything unrecognised is ignored.
    #[test]
    fn the_route_override_forces_either_path() {
        let r = get_base_range_u128(57).unwrap().unwrap();
        let s = r.range_start;
        let small = FieldSize::new(s, s + 4_000_000_000);
        let field = FieldSize::new(s, s + JOIN_MIN_FIELD_SIZE);
        assert_eq!(
            join_verdict_routed(57, &field, RouteOverride::Stride),
            Err(StrideReason::Forced)
        );
        assert!(join_verdict_routed(57, &small, RouteOverride::Join).is_ok());
        let r = get_base_range_u128(35).unwrap().unwrap();
        let tiny = FieldSize::new(r.range_start, r.range_start + 1_000);
        assert_eq!(
            join_verdict_routed(35, &tiny, RouteOverride::Join),
            Err(StrideReason::Base)
        );
        let edge = 50u128.pow(10);
        assert_eq!(
            join_verdict_routed(50, &FieldSize::new(edge - 5, edge + 5), RouteOverride::Join),
            Err(StrideReason::LengthChange)
        );
        for (value, route) in [
            ("", Some(RouteOverride::Auto)),
            ("auto", Some(RouteOverride::Auto)),
            ("JOIN", Some(RouteOverride::Join)),
            (" stride\n", Some(RouteOverride::Stride)),
            ("both", None),
        ] {
            assert_eq!(RouteOverride::parse(value), route, "{value:?}");
        }
    }

    #[test]
    fn div64_matches_u128_division() {
        let mut x = 0x9E37_79B9_7F4A_7C15_u128 * 0xBF58_476D_1CE4_E5B9;
        for d in [3u64, 57, 185_193, 1 << 40, u64::MAX - 58] {
            let dv = Div64::new(d);
            for _ in 0..200 {
                x = x
                    .wrapping_mul(0x5851_F42D_4C95_7F2D)
                    .wrapping_add(1_442_695_040_888_963_407);
                let (q, r) = dv.divrem_u128(x);
                assert_eq!((q, u128::from(r)), (x / u128::from(d), x % u128::from(d)));
            }
        }
    }
}
