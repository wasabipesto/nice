//! The overlap join on the CPU. [`crate::overlap_join`] explains the join
//! and holds the reference it is tested against; `crate::cubecl_join` is
//! the same join on a GPU.
//!
//! A field is `b^p` partitions, independent of each other, which the client
//! runs in parallel ([`CpuJoin::run_partition`]). One partition:
//!
//! 1. **Tops**, one key digit at a time: the key's top-layer prefixes
//!    extended by the partition's digits and certified, as in the reference
//!    join. A thread holds one key's tops (about `1/b` of the partition's).
//!    A key's prefixes come in ascending order, a multiple of a power of the
//!    base apart, so their certificates step from one to the next by
//!    digit-array additions (`cpu_top_fd`) instead of dividing.
//! 2. **Bottoms**: every residue of the field's bottom list extended by the
//!    partition's fixed digits, with exact arithmetic once per residue.
//!    What the last bottom digit `d` (the key digit) adds is affine in `d`:
//!    the output digits at position `k − 1`, and the prefilter's above it,
//!    come from `(q + d·c) mod b^(1 + mid)`.
//! 3. **Join**, one key digit at a time: the key's bottoms are listed by
//!    digit-sum class (about a megabyte at base 57, so they stay in cache),
//!    and each of the key's tops scans the classes its roots name, one
//!    64-bit AND per bottom.
//! 4. **Survivors** go through the middle-digit prefilter, as on the GPU
//!    (its digits read off a fixed-point fraction), and then the client's
//!    own full check.
//!
//! The client sends it the nice-only fields the GPU's join would take, a
//! slice at a time ([`slices_for`]): the host's top layer grows with the
//! field, so a field with more than [`CPU_SLICE_PREFIXES`] top-layer
//! prefixes is cut into slices of that many. Its survivors, and the
//! prefilter's, are exactly the reference join's: the tests compare them
//! partition by partition.

use crate::FieldSize;
use crate::client_process::{get_is_nice, get_is_nice_with_known_lsd};
use crate::cpu_top_fd::{TopFd, TopFdSetup};
use crate::overlap_join::{
    Div64, FieldSetup, JoinParams, join_params_for, join_slices, join_verdict, ndigits,
    prefix_block,
};
use anyhow::Result;

/// The most top-layer prefixes one slice of a field may have on the CPU
/// (see [`slices_for`]). This bounds the host's top layer (about 32 bytes a
/// prefix, so 0.5 GiB at most) and its key index; a thread's tops are one
/// key's share of it. Production fields are one slice up to about 1e16.
pub const CPU_SLICE_PREFIXES: u128 = 1 << 24;

/// A nice-only field the CPU join takes (the GPU's gate,
/// `overlap_join::join_verdict`), as the slices it runs in: its parameters
/// and the slices, in order. `None` if it takes the stride walk.
#[must_use]
pub fn slices_for(base: u32, range: &FieldSize) -> Option<(JoinParams, Vec<FieldSize>)> {
    let jp = join_verdict(base, range).ok()?;
    let block = prefix_block(base, ndigits(range.last(), base), jp);
    Some((jp, join_slices(range, block, CPU_SLICE_PREFIXES)))
}

/// The mask of a bottom whose fixed digits repeat: it joins no top.
const DEAD: u64 = u64::MAX;

/// What one partition through the join found and did.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PartitionResult {
    /// The partition's nice numbers.
    pub hits: Vec<u128>,
    /// Pairs that passed the join's AND inside the field, and of those the
    /// ones that passed the prefilter (which the full check read).
    pub survivors: u64,
    pub checked: u64,
}

impl PartitionResult {
    /// Add another partition's result to this one.
    pub fn add(&mut self, other: PartitionResult) {
        self.hits.extend(other.hits);
        self.survivors += other.survivors;
        self.checked += other.checked;
    }
}

/// Buffers a thread reuses from one partition to the next.
#[derive(Default)]
pub struct Scratch {
    tops: Vec<Top>,
    ext: Ext,
    masks: Vec<u64>,
    info: Vec<Info>,
    offs: Vec<u32>,
}

/// A top prefix of the partition: `n = P·b^f0 + r` for the bottom residues
/// `r` (mod `b^f0`) in `rlo..rhi`.
#[derive(Clone, Copy)]
#[cfg_attr(test, derive(Debug, PartialEq, Eq))]
struct Top {
    p: u128,
    /// The certificate: output digits at positions `>= k`.
    mask: u64,
    /// The certificate if every certified position is `>= k2`, so that the
    /// prefilter's digits must avoid it too; otherwise 0.
    gmask: u64,
    rlo: u32,
    rhi: u32,
    /// `⌈D·2^32/b²⌉` for `D` = `n`'s digits `k..k2` (the next digits of
    /// `P`) as a number: the prefilter's fixed-point `D` ([`Fx`]).
    dh: u32,
    /// `P mod (b − 1)`.
    pc: u32,
}

// The tops a thread sorts and scans: 48 bytes each (`u128` alignment).
const _: () = assert!(std::mem::size_of::<Top>() == 48);

/// The bottom-list residues that survive the partition's fixed digits,
/// each `r` with what the last bottom digit `d` needs, by column: the
/// running `⌊R²/b^(k−1)⌋` and `⌊R³/b^(k−1)⌋` mod `M = b^(1 + mid)` for
/// `R = r + d·b^(k−1)`, and their steps in `d`.
#[derive(Default)]
struct Ext {
    /// The residue's index in the field's bottom list.
    idx: Vec<u32>,
    /// Output digits at positions below `k − 1`.
    mask: Vec<u64>,
    /// `⌊r²/b^(k−1)⌋` and `⌊r³/b^(k−1)⌋` mod `M` (the values at `d = 0`).
    q2: Vec<u32>,
    q3: Vec<u32>,
    /// The values at the current `d`.
    t2: Vec<u32>,
    t3: Vec<u32>,
    /// `2r`, `3r²` and `3r·b^(k−1)` mod `M` (the last is zero for the
    /// production parameters).
    c2: Vec<u32>,
    c3: Vec<u32>,
    c3b: Vec<u32>,
    /// `r mod b^mid`, and the prefilter's coefficients of the top's digits
    /// `2r` and `3r²` mod `b^mid` (valid when the last digit does not reach
    /// below `b^mid`, as for the production parameters).
    rho: Vec<u32>,
    p2: Vec<u32>,
    p3: Vec<u32>,
    /// Where each digit-sum class starts, as in `FieldSetup::seg`.
    seg: Vec<u32>,
    /// The last digit `t2`/`t3` hold.
    d: u32,
}

impl Ext {
    fn clear(&mut self) {
        for v in [
            &mut self.idx,
            &mut self.q2,
            &mut self.q3,
            &mut self.t2,
            &mut self.t3,
            &mut self.c2,
            &mut self.c3,
            &mut self.c3b,
            &mut self.rho,
            &mut self.p2,
            &mut self.p3,
            &mut self.seg,
        ] {
            v.clear();
        }
        self.mask.clear();
        self.d = 0;
    }
}

/// A listed bottom: its row in [`Ext`], and the carries the prefilter
/// needs, `⌊R²/b^k⌋ | ⌊R³/b^k⌋ << 16` (each mod `b^mid`, below `2^12`).
#[derive(Clone, Copy)]
struct Info {
    row: u32,
    carries: u32,
}

/// Exact division by a small divisor through a multiply: for `x < 2^48/d`
/// ([`Self::divrem`]), or `x < 2^26` and `d <= 64` ([`Self::divrem_small`]).
#[derive(Clone, Copy)]
pub(crate) struct SmallDiv {
    d: u64,
    magic: u64,
    magic32: u64,
}

// The quotients and remainders are below the bounds the methods state, and
// `d` below 2^32 wherever it is narrowed.
#[allow(clippy::cast_possible_truncation)]
impl SmallDiv {
    pub(crate) fn new(d: u64) -> Self {
        Self {
            d,
            magic: (1u64 << 48).div_ceil(d),
            magic32: (1u64 << 32).div_ceil(d),
        }
    }

    /// `(x / d, x % d)`.
    #[inline]
    pub(crate) fn divrem(self, x: u64) -> (u64, u64) {
        let q = ((u128::from(x) * u128::from(self.magic)) >> 48) as u64;
        (q, x - q * self.d)
    }

    /// `(x / d, x % d)` with one 64-bit multiply, exact below `2^32/d`
    /// (which `2^26` is for `d <= 64`).
    #[inline]
    fn divrem_small(self, x: u32) -> (u32, u32) {
        let q = ((u64::from(x) * self.magic32) >> 32) as u32;
        (q, x - q * self.d as u32)
    }
}

/// The prefilter's two digits at `mid = 2` by fixed point: the base-b
/// digits `(h, g)` of `x = (A + P·D) mod b²` for `A, P, D < b²` and `b <=
/// 64`, with four multiplies (`A·C`, `P·δ̂` and two by `b`) and no
/// reduction.
///
/// With `C = ⌈2^32/b²⌉` and `δ̂ = ⌈D·2^32/b²⌉`, write `C·b² = 2^32 + ε1`
/// and `δ̂·b² = D·2^32 + ε2` (`0 <= ε1, ε2 < b²`), and `A + P·D = m·b² +
/// x`. Then `(A·C + P·δ̂)·b² = m·b²·2^32 + x·2^32 + Δ` with `Δ = A·ε1 +
/// P·ε2 < 2b⁴ <= 2^25`. So `F = (A·C + P·δ̂) mod 2^32` is `(x·2^32 + Δ)/b²`,
/// an integer below 2^32, and `F·b² = x·2^32 + Δ`. Two multiplies by `b`
/// split `F·b²` exactly, `F·b = h·2^32 + F'` and `F'·b = g·2^32 + F''`
/// (`g < b`), so `h·b + g = ⌊F·b²/2^32⌋`, which is `x` because `Δ <
/// 2^32`.
#[derive(Clone, Copy)]
struct Fx {
    b: u64,
    /// `C = ⌈2^32/b²⌉`.
    c: u32,
    /// `2^32 = q·b² + r`, for `δ̂ = D·q + ⌈D·r/b²⌉`.
    q: u32,
    r: u32,
    b2: SmallDiv,
}

// Narrowed: C and q are at most 2^32/b² + 1 and r below b², for b >= 2.
#[allow(clippy::cast_possible_truncation)]
impl Fx {
    fn new(b: u32) -> Self {
        let b2 = u64::from(b) * u64::from(b);
        Self {
            b: u64::from(b),
            c: (1u64 << 32).div_ceil(b2) as u32,
            q: ((1u64 << 32) / b2) as u32,
            r: ((1u64 << 32) % b2) as u32,
            b2: SmallDiv::new(b2),
        }
    }

    /// `δ̂ = ⌈D·2^32/b²⌉` for `D < b²` (once per top). `D·r + b² − 1 <
    /// b⁴ <= 2^24` is in `SmallDiv::divrem`'s range, and the sum is below
    /// `2^32`.
    fn dh(self, d: u32) -> u32 {
        let (up, _) = self
            .b2
            .divrem(u64::from(d) * u64::from(self.r) + self.b2.d - 1);
        d * self.q + up as u32
    }

    /// `x`'s digits `(h, g)`, high and low, from `A`, `P` and `δ̂`. `A·C`
    /// fits a word: `(b² − 1)·C < 2^32` for `b < 256`.
    #[inline]
    fn digits(self, a: u32, p: u32, dh: u32) -> (u64, u64) {
        let f = (a * self.c).wrapping_add(p.wrapping_mul(dh));
        let fb = u64::from(f) * self.b;
        (fb >> 32, (u64::from(fb as u32) * self.b) >> 32)
    }

    /// The prefilter at `mid = 2`: the digits of `x2 = (A2 + P2·D) mod b²`
    /// and `x3 = (A3 + P3·D) mod b²` (`carries = A2 | A3 << 16`, `D` as
    /// `δ̂`) pairwise distinct and none in `seen` ([`fresh4`]).
    #[inline]
    fn pass(self, seen: u64, carries: u32, p2: u32, p3: u32, dh: u32) -> bool {
        let (h2, g2) = self.digits(carries & 0xFFFF, p2, dh);
        let (h3, g3) = self.digits(carries >> 16, p3, dh);
        fresh4(seen, h2, g2, h3, g3)
    }
}

/// Four digits (each below 64) pairwise distinct and none in `seen`: one
/// shift per digit and six compares, without a branch. The same predicate
/// as [`fresh_digits`]' four one-hot masks at `mid = 2`.
// `&`, not `&&`: branch-free on purpose (the outcome is data-dependent).
#[allow(clippy::needless_bitwise_bool)]
#[inline]
fn fresh4(seen: u64, h2: u64, g2: u64, h3: u64, g3: u64) -> bool {
    let hit = (seen >> g2) | (seen >> g3) | (seen >> h2) | (seen >> h3);
    ((hit & 1) == 0) & (g2 != g3) & (h2 != h3) & (g2 != h2) & (g2 != h3) & (g3 != h2) & (g3 != h3)
}

/// The prefilter's digits of `n²` and `n³` (`x2`, `x3`, `mid` digits each,
/// below `b^mid`), divided out low digit first: distinct from each other
/// and new against `seen` (the bottom's digits, and the certificate when
/// it sits above them).
#[inline]
fn fresh_digits(sb: SmallDiv, mid: u32, mut seen: u64, mut x2: u32, mut x3: u32) -> bool {
    for _ in 0..mid {
        let (q2, g2) = sb.divrem_small(x2);
        let (q3, g3) = sb.divrem_small(x3);
        let bits = (1u64 << g2) | (1u64 << g3);
        if g2 == g3 || seen & bits != 0 {
            return false;
        }
        seen |= bits;
        (x2, x3) = (q2, q3);
    }
    true
}

/// A field prepared for the CPU join.
pub struct CpuJoin {
    fs: FieldSetup,
    k: u32,
    /// Prefilter digits above `k` (`k2 − k`).
    mid: u32,
    /// Values of the last bottom digit: `b` when it is the free key digit;
    /// otherwise it is the partition's last digit, and there is one.
    keyspace: u32,
    /// `M = b^(1 + mid)` and `b^mid`.
    big_m: u64,
    bmid: u64,
    /// `b^(k−1) mod M` and `b^(2(k−1)) mod M`. Zero for the production
    /// parameters, which then take the affine path.
    e1: u64,
    e2: u64,
    /// `b^(k−1)`, `b^k` and `b^(2k)` mod `b^mid`: zero unless `k` is below
    /// the prefilter's depth (small test parameters).
    kb: u64,
    bk: u64,
    b2k: u64,
    /// `b^(k − f0)`: from `P` down to `n`'s digit `k`.
    pk: u128,
    div_k1: Div64,
    div_m: Div64,
    sb: SmallDiv,
    /// The prefilter's digits by fixed point, and whether the field takes
    /// them: `mid = 2` and `k > mid`, so that the last bottom digit adds
    /// nothing below `b^mid` (every production shape). Every other shape
    /// takes the general form ([`Self::prefilter_any`]).
    fx: Fx,
    lean: bool,
    /// The top certificates by finite differences, for every unclipped
    /// block; `None` if the field's digit arrays would not fit them (then
    /// `Base::cert` certifies every block).
    fd: Option<TopFdSetup>,
    /// The AND scan may use AVX2 (detected at run time).
    #[cfg(target_arch = "x86_64")]
    avx2: bool,
    /// The AND scan may use AVX-512F (detected at run time).
    #[cfg(target_arch = "x86_64")]
    avx512: bool,
    /// `fs.tlay` is ordered by key digit (stably): `key_off[d]..key_off[d +
    /// 1]` are the prefixes whose key is `d`.
    key_off: Vec<u32>,
}

impl CpuJoin {
    /// The CPU join for a nice-only field as one piece, if it takes the
    /// field: the same gate as the GPU's (`overlap_join::join_params_for`).
    /// The client runs fields a slice at a time ([`slices_for`]).
    #[must_use]
    pub fn for_field(base: u32, range: &FieldSize) -> Option<Self> {
        let jp = join_params_for(base, range)?;
        Self::new(base, range, jp).ok()
    }

    /// Set up `range` at `base` with parameters `jp`.
    ///
    /// # Errors
    /// A field or parameters the join cannot take (see `FieldSetup::new`).
    pub fn new(base: u32, range: &FieldSize, jp: JoinParams) -> Result<Self> {
        let mut fs = FieldSetup::new(base, range.start(), range.end(), jp)?;
        let b = u64::from(base);
        let k = jp.k;
        let mid = fs.k2 - k;
        let big_m = b.pow(1 + mid);
        let bmid = b.pow(mid);
        let bk1 = b.pow(k - 1);
        let keyspace = if fs.key_level { base } else { 1 };
        // The top layer by key digit (stably), so a key's prefixes are
        // contiguous; the join only reads them a key at a time.
        // A key digit is below b <= 64.
        #[allow(clippy::cast_possible_truncation)]
        let key_of = |p0: u128| (p0 % u128::from(keyspace)) as usize;
        fs.tlay.sort_by_key(|&(p0, _)| key_of(p0));
        let mut key_off = vec![0u32; keyspace as usize + 1];
        for &(p0, _) in &fs.tlay {
            key_off[key_of(p0) + 1] += 1;
        }
        for d in 0..keyspace as usize {
            key_off[d + 1] += key_off[d];
        }
        // A key's prefixes are `b^(p + key level) = b^o` apart in P, and
        // `o = k − f0`: the step is `pk`.
        let fd = TopFdSetup::try_new(&fs.base, fs.f0, k - fs.f0, k, fs.phi, u32::try_from(bmid)?);
        Ok(Self {
            k,
            mid,
            keyspace,
            key_off,
            big_m,
            bmid,
            e1: bk1 % big_m,
            e2: u64::try_from(u128::from(bk1) * u128::from(bk1) % u128::from(big_m))?,
            kb: bk1 % bmid,
            bk: bk1 * b % bmid,
            b2k: u64::try_from(u128::from(bk1 * b).pow(2) % u128::from(bmid))?,
            pk: fs.base.powu(k - fs.f0),
            div_k1: Div64::new(bk1),
            div_m: Div64::new(big_m),
            sb: SmallDiv::new(b),
            fx: Fx::new(base),
            lean: mid == 2 && k > mid,
            fd,
            #[cfg(target_arch = "x86_64")]
            avx2: std::arch::is_x86_feature_detected!("avx2"),
            #[cfg(target_arch = "x86_64")]
            avx512: std::arch::is_x86_feature_detected!("avx512f"),
            fs,
        })
    }

    /// Partitions in the field (`b^p`); each is a [`Self::run_partition`].
    ///
    /// # Panics
    /// Never: `b^p` fits `u32` for every supported parameter set.
    #[must_use]
    pub fn partitions(&self) -> u32 {
        u32::try_from(self.fs.nparts).expect("b^p fits u32")
    }

    /// The host setup's time, seconds.
    #[must_use]
    pub fn setup_secs(&self) -> f64 {
        self.fs.secs
    }

    /// Partition `v` of the field.
    #[must_use]
    pub fn run_partition(&self, v: u32, scratch: &mut Scratch) -> PartitionResult {
        self.run(v, scratch, None)
    }

    /// Every partition, on `threads` threads of its own (the client runs
    /// partitions on its pool instead).
    ///
    /// # Panics
    /// If a partition does (a bug).
    #[must_use]
    pub fn run_all(&self, threads: usize) -> PartitionResult {
        use std::sync::atomic::{AtomicU32, Ordering};
        let next = AtomicU32::new(0);
        let parts = self.partitions();
        std::thread::scope(|s| {
            let workers: Vec<_> = (0..threads.max(1))
                .map(|_| {
                    s.spawn(|| {
                        let mut scratch = Scratch::default();
                        let mut out = PartitionResult::default();
                        loop {
                            let v = next.fetch_add(1, Ordering::Relaxed);
                            if v >= parts {
                                return out;
                            }
                            out.add(self.run_partition(v, &mut scratch));
                        }
                    })
                })
                .collect();
            let mut out = PartitionResult::default();
            for w in workers {
                out.add(w.join().expect("a partition worker panicked"));
            }
            out.hits.sort_unstable();
            out
        })
    }

    /// Digit `i` of the partition value `v`.
    fn partition_digit(&self, v: u32, i: u32) -> u32 {
        (v / self.fs.b.pow(i)) % self.fs.b
    }

    fn run(&self, v: u32, sc: &mut Scratch, mut record: Option<&mut Vec<u128>>) -> PartitionResult {
        let mut out = PartitionResult::default();
        let m1 = self.fs.m1;
        let roots = &self.fs.base.roots;
        // The bottoms are extended at the first key with a top, so a
        // partition without tops does no bottom work at all.
        let mut extended = false;
        for key in 0..self.keyspace {
            let (lo, hi) = (
                self.key_off[key as usize] as usize,
                self.key_off[key as usize + 1] as usize,
            );
            if lo == hi {
                continue;
            }
            self.tops(v, &self.fs.tlay[lo..hi], &mut sc.tops);
            if sc.tops.is_empty() {
                continue;
            }
            if !extended {
                self.extend(v, &mut sc.ext);
                extended = true;
            }
            sc.tops.sort_unstable_by_key(|t| t.pc);
            // The last bottom digit: the key itself, or the partition's last.
            let d = if self.fs.key_level {
                key
            } else {
                self.partition_digit(v, self.fs.jp.p - 1)
            };
            self.list(d, &mut sc.ext, &mut sc.masks, &mut sc.info, &mut sc.offs);
            for top in &sc.tops {
                for &root in roots {
                    let c = ((root + m1 - top.pc) % m1) as usize;
                    let (lo, hi) = (sc.offs[c] as usize, sc.offs[c + 1] as usize);
                    let pass = Pass {
                        top,
                        d,
                        ext: &sc.ext,
                        info: &sc.info[lo..hi],
                    };
                    self.scan(&pass, &sc.masks[lo..hi], &mut out, &mut record);
                }
            }
        }
        out
    }

    /// The partition's tops from the top-layer prefixes `layer` (all of one
    /// key, ascending), as the reference join makes them. An unclipped
    /// block's certificate steps from the last one's (`TopFdSetup`); a
    /// clipped one, at either end of the field, is divided out.
    // Below 2^32: w = b^f0 is (`JoinParams::supported`), and so are b^mid,
    // b - 1 and the key space. w, p, a, e as in `overlap_join`.
    #[allow(clippy::cast_possible_truncation, clippy::many_single_char_names)]
    fn tops(&self, v: u32, layer: &[(u128, u64)], tops: &mut Vec<Top>) {
        tops.clear();
        let fs = &self.fs;
        let (w, v) = (fs.w, u128::from(v));
        let mut st = TopFd::default();
        let full_seed = fs.full_floor >= fs.k2;
        for &(p0, _) in layer {
            let p = p0 * fs.nparts + v;
            if p < fs.plo || p > fs.phi {
                continue;
            }
            let lo = p * w;
            let a = lo.max(fs.s);
            let e = (lo + w - 1).min(fs.e - 1);
            if let Some(fd) = &self.fd
                && a == lo
                && e == lo + w - 1
            {
                fd.advance(&mut st, p);
                if let Some(mask) = fd.cert(&st) {
                    tops.push(Top {
                        p,
                        mask,
                        gmask: if full_seed { mask } else { 0 },
                        rlo: 0,
                        rhi: w as u32,
                        dh: self.fx.dh(st.dm),
                        pc: st.pc,
                    });
                }
                continue;
            }
            let Some(mask) = fs.base.cert(a, e, self.k) else {
                continue;
            };
            let floor = if a == lo && e == lo + w - 1 {
                fs.full_floor
            } else {
                fs.base.cert_floor(a, e, self.k)
            };
            tops.push(Top {
                p,
                mask,
                gmask: if floor >= fs.k2 { mask } else { 0 },
                rlo: (a - lo) as u32,
                rhi: (e - lo + 1) as u32,
                dh: self.fx.dh(((p / self.pk) % u128::from(self.bmid)) as u32),
                pc: (p % u128::from(fs.m1)) as u32,
            });
        }
    }

    /// The bottom list extended by the partition's fixed digits: the
    /// positions from `f0` up to `k − 2` (`k − 1` holds the key digit, or
    /// the partition's last digit, which [`Self::list`] adds). Residues
    /// whose digits repeat, or meet the field's certificate (which every
    /// top's mask contains), are dropped here, once for every key.
    // Narrowed: b^f0 and the list's length (at most b^f0) are below 2^32,
    // r below b^(k-1) < 2^40, and every column below M <= 2^18.
    #[allow(clippy::cast_possible_truncation)]
    fn extend(&self, v: u32, ext: &mut Ext) {
        let fs = &self.fs;
        let (b, k, f0) = (u64::from(fs.b), self.k, fs.f0);
        let fixed = k - 1 - f0;
        let mut add = 0u128;
        for i in 0..fixed {
            add += u128::from(self.partition_digit(v, i)) * fs.base.powu(f0 + i);
        }
        let below_f0 = fs.base.powu(f0) as u64;
        let fc = fs.fc;
        ext.clear();
        for c in 0..fs.m1 as usize {
            ext.seg.push(ext.idx.len() as u32);
            for idx in fs.seg[c]..fs.seg[c + 1] {
                let (r0, m0) = (fs.bp_r[idx as usize], fs.bp_m[idx as usize]);
                let r = u128::from(r0) + add;
                let (r2, r3) = (r * r, r * r * r);
                let (q2, lo2) = self.div_k1.divrem_u128(r2);
                let (q3, lo3) = self.div_k1.divrem_u128(r3);
                // Output digits at positions f0..k-1 of r² and r³: fixed now.
                let mut mask = m0;
                let (mut x2, mut x3) = (lo2 / below_f0, lo3 / below_f0);
                for _ in 0..fixed {
                    let (g2, g3) = (x2 % b, x3 % b);
                    x2 /= b;
                    x3 /= b;
                    let bits = (1u64 << g2) | (1u64 << g3);
                    if g2 == g3 || (mask | fc) & bits != 0 {
                        mask = DEAD;
                        break;
                    }
                    mask |= bits;
                }
                if mask == DEAD {
                    continue;
                }
                let r = r as u64;
                let rm = r % self.big_m;
                let (q2, q3) = (
                    self.div_m.divrem_u128(q2).1 as u32,
                    self.div_m.divrem_u128(q3).1 as u32,
                );
                ext.idx.push(idx);
                ext.mask.push(mask);
                ext.q2.push(q2);
                ext.q3.push(q3);
                ext.t2.push(q2);
                ext.t3.push(q3);
                ext.c2.push((2 * rm % self.big_m) as u32);
                ext.c3
                    .push((3 * (rm * rm % self.big_m) % self.big_m) as u32);
                ext.c3b
                    .push((3 * rm % self.big_m * self.e1 % self.big_m) as u32);
                let rho = r % self.bmid;
                ext.rho.push(rho as u32);
                ext.p2.push((2 * rho % self.bmid) as u32);
                ext.p3
                    .push((3 * (rho * rho % self.bmid) % self.bmid) as u32);
            }
        }
        ext.seg.push(ext.idx.len() as u32);
    }

    /// Move every residue's `t2`/`t3` to the last bottom digit `d`. For the
    /// production parameters they are affine in `d`: one step is an
    /// addition mod `M` per residue, over plain columns.
    // M <= 2^18, so it and everything reduced mod M fit u32.
    #[allow(clippy::cast_possible_truncation)]
    fn advance(&self, ext: &mut Ext, d: u32) {
        let m = self.big_m as u32;
        if self.e1 == 0 && self.e2 == 0 && d >= ext.d && d - ext.d <= 2 {
            for _ in ext.d..d {
                for (t, &c) in ext.t2.iter_mut().zip(&ext.c2) {
                    let x = *t + c;
                    *t = if x >= m { x - m } else { x };
                }
                for (t, &c) in ext.t3.iter_mut().zip(&ext.c3) {
                    let x = *t + c;
                    *t = if x >= m { x - m } else { x };
                }
            }
        } else {
            let (d64, m64) = (u64::from(d), self.big_m);
            for i in 0..ext.idx.len() {
                ext.t2[i] =
                    ((u64::from(ext.q2[i]) + d64 * u64::from(ext.c2[i]) + d64 * d64 * self.e1)
                        % m64) as u32;
                ext.t3[i] = ((u64::from(ext.q3[i])
                    + d64 * u64::from(ext.c3[i])
                    + d64 * d64 * u64::from(ext.c3b[i])
                    + d64 * d64 * d64 * self.e2)
                    % m64) as u32;
            }
        }
        ext.d = d;
    }

    /// The bottoms whose last digit is `d`, listed by digit-sum class:
    /// `masks` and `info` side by side, class `c` at `offs[c]..offs[c + 1]`.
    // A list is at most the bottom list, below b^f0 < 2^32 entries.
    #[allow(clippy::cast_possible_truncation)]
    fn list(
        &self,
        d: u32,
        ext: &mut Ext,
        masks: &mut Vec<u64>,
        info: &mut Vec<Info>,
        offs: &mut Vec<u32>,
    ) {
        self.advance(ext, d);
        masks.clear();
        info.clear();
        offs.clear();
        let fc = self.fs.fc;
        for c in 0..self.fs.m1 as usize {
            offs.push(masks.len() as u32);
            for row in ext.seg[c]..ext.seg[c + 1] {
                let i = row as usize;
                let (a2, g2) = self.sb.divrem_small(ext.t2[i]);
                let (a3, g3) = self.sb.divrem_small(ext.t3[i]);
                let bits = (1u64 << g2) | (1u64 << g3);
                let mask = ext.mask[i];
                if g2 == g3 || (mask | fc) & bits != 0 {
                    continue;
                }
                masks.push(mask | bits);
                info.push(Info {
                    row,
                    carries: a2 | (a3 << 16),
                });
            }
        }
        offs.push(masks.len() as u32);
    }

    /// One top against one class of bottoms: a 64-bit AND each, eight at a
    /// time; the (rare) pairs that pass go to [`Self::survivor`].
    fn scan(
        &self,
        pass: &Pass,
        masks: &[u64],
        out: &mut PartitionResult,
        record: &mut Option<&mut Vec<u128>>,
    ) {
        #[cfg(target_arch = "x86_64")]
        if self.avx512 {
            // SAFETY: `avx512` is only set where the CPU has AVX-512F.
            unsafe { self.scan_avx512(pass, masks, out, record) };
            return;
        }
        #[cfg(target_arch = "x86_64")]
        if self.avx2 {
            // SAFETY: `avx2` is only set where the CPU has AVX2.
            unsafe { self.scan_avx2(pass, masks, out, record) };
            return;
        }
        self.scan_any(pass, masks, out, record);
    }

    /// [`Self::scan_any`] with AVX-512F: `vptestnmq` gives the zero-AND
    /// lanes of eight bottoms as a mask directly; 32 per step.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    fn scan_avx512(
        &self,
        pass: &Pass,
        masks: &[u64],
        out: &mut PartitionResult,
        record: &mut Option<&mut Vec<u128>>,
    ) {
        use std::arch::x86_64::{_mm512_loadu_si512, _mm512_set1_epi64, _mm512_testn_epi64_mask};
        let t = _mm512_set1_epi64(pass.top.mask.cast_signed());
        let (chunks, rest) = masks.as_chunks::<32>();
        let mut at = 0;
        for chunk in chunks {
            let mut bits = 0u32;
            for q in 0..4 {
                // SAFETY: eight u64 at q * 8 of a 32-element chunk; the load
                // is unaligned (`loadu`).
                let v = unsafe { _mm512_loadu_si512(chunk.as_ptr().add(8 * q).cast()) };
                bits |= u32::from(_mm512_testn_epi64_mask(v, t)) << (8 * q);
            }
            while bits != 0 {
                let j = at + bits.trailing_zeros() as usize;
                bits &= bits - 1;
                self.survivor(pass, masks[j], j, out, record);
            }
            at += 32;
        }
        let t = pass.top.mask;
        for (j, &m) in rest.iter().enumerate() {
            if m & t == 0 {
                self.survivor(pass, m, at + j, out, record);
            }
        }
    }

    /// [`Self::scan_any`] with AVX2: sixteen ANDs per step, compared with
    /// zero and turned into a bit per bottom.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    fn scan_avx2(
        &self,
        pass: &Pass,
        masks: &[u64],
        out: &mut PartitionResult,
        record: &mut Option<&mut Vec<u128>>,
    ) {
        use std::arch::x86_64::{
            __m256i, _mm256_and_si256, _mm256_castsi256_pd, _mm256_cmpeq_epi64, _mm256_loadu_si256,
            _mm256_movemask_pd, _mm256_set1_epi64x, _mm256_setzero_si256,
        };
        let t = _mm256_set1_epi64x(pass.top.mask.cast_signed());
        let zero = _mm256_setzero_si256();
        let (chunks, rest) = masks.as_chunks::<16>();
        let mut at = 0;
        for chunk in chunks {
            let mut bits = 0u32;
            for q in 0..4 {
                // SAFETY: four u64 at q * 4 of a 16-element chunk; the load
                // is unaligned (`loadu`), so the pointer cast is too.
                #[allow(clippy::cast_ptr_alignment)]
                let v = unsafe { _mm256_loadu_si256(chunk.as_ptr().add(4 * q).cast::<__m256i>()) };
                let z = _mm256_cmpeq_epi64(_mm256_and_si256(v, t), zero);
                bits |= (_mm256_movemask_pd(_mm256_castsi256_pd(z)).cast_unsigned()) << (4 * q);
            }
            while bits != 0 {
                let j = at + bits.trailing_zeros() as usize;
                bits &= bits - 1;
                self.survivor(pass, masks[j], j, out, record);
            }
            at += 16;
        }
        let t = pass.top.mask;
        for (j, &m) in rest.iter().enumerate() {
            if m & t == 0 {
                self.survivor(pass, m, at + j, out, record);
            }
        }
    }

    /// The scan without AVX2: eight ANDs per step.
    fn scan_any(
        &self,
        pass: &Pass,
        masks: &[u64],
        out: &mut PartitionResult,
        record: &mut Option<&mut Vec<u128>>,
    ) {
        let t = pass.top.mask;
        let (chunks, rest) = masks.as_chunks::<8>();
        let mut at = 0;
        for chunk in chunks {
            let mut bits = 0u32;
            for (j, &m) in chunk.iter().enumerate() {
                bits |= u32::from(m & t == 0) << j;
            }
            while bits != 0 {
                let j = at + bits.trailing_zeros() as usize;
                bits &= bits - 1;
                self.survivor(pass, masks[j], j, out, record);
            }
            at += 8;
        }
        for (j, &m) in rest.iter().enumerate() {
            if m & t == 0 {
                self.survivor(pass, m, at + j, out, record);
            }
        }
    }

    /// A pair that passed the AND: inside the field, through the prefilter,
    /// then the full check.
    #[inline(never)]
    fn survivor(
        &self,
        pass: &Pass,
        bmask: u64,
        j: usize,
        out: &mut PartitionResult,
        record: &mut Option<&mut Vec<u128>>,
    ) {
        let info = pass.info[j];
        let row = info.row as usize;
        let r0 = self.fs.bp_r[pass.ext.idx[row] as usize];
        let top = pass.top;
        if r0 < top.rlo || r0 >= top.rhi {
            return;
        }
        out.survivors += 1;
        let ext = pass.ext;
        let pass_mid = if self.lean {
            self.prefilter(top, bmask, info.carries, ext.p2[row], ext.p3[row])
        } else {
            // R mod b^mid: the residue's, plus what the last digit adds.
            let rho = u64::from(ext.rho[row]) + u64::from(pass.d) * self.kb;
            self.prefilter_any(top, bmask, info.carries, rho % self.bmid)
        };
        if !pass_mid {
            return;
        }
        out.checked += 1;
        let n = top.p * self.fs.w + u128::from(r0);
        if let Some(rec) = record.as_deref_mut() {
            rec.push(n);
        }
        // With f0 = 3 (the production parameters) the bottom list's own
        // mask is the low three digits of both powers, which the check skips.
        let nice = if self.fs.f0 == 3 {
            get_is_nice_with_known_lsd(n, self.fs.b, 3, self.fs.bp_m[ext.idx[row] as usize])
        } else {
            get_is_nice(n, self.fs.b)
        };
        if nice {
            out.hits.push(n);
        }
    }

    /// The middle-digit prefilter: `n`'s output digits at positions
    /// `k..k2` must be distinct and new (and avoid the certificate when it
    /// sits above them). With `n = R + D·b^k` (`R` the bottom, `D` the top's
    /// next digits) they are the digits of `⌊R²/b^k⌋ + 2RD` and
    /// `⌊R³/b^k⌋ + 3R²D` mod `b^mid` when `k > mid` (the production
    /// parameters), with `2R`, `3R²` (`p2`, `p3`) the bottom's. At `mid = 2`
    /// (`self.lean`) the digits come by fixed point ([`Fx`]), the top's `D`
    /// as `δ̂`, and its certificate is pre-gated (`gmask`).
    #[inline]
    fn prefilter(&self, top: &Top, bmask: u64, carries: u32, p2: u32, p3: u32) -> bool {
        self.fx.pass(bmask | top.gmask, carries, p2, p3, top.dh)
    }

    /// [`Self::prefilter`] for any parameters: also the terms in `b^k`
    /// (`D²b^k`, `3RD²b^k`, `D³b^(2k)`), which vanish unless `k` is below
    /// the prefilter's depth, and `R mod b^mid` as given.
    // D = P's digits k..k2 as a number, below b^mid < 2^32.
    #[allow(clippy::cast_possible_truncation)]
    fn prefilter_any(&self, top: &Top, bmask: u64, carries: u32, rho: u64) -> bool {
        let m = self.bmid;
        let dm = ((top.p / self.pk) % u128::from(m)) as u64;
        let (a2, a3) = (u64::from(carries & 0xFFFF), u64::from(carries >> 16));
        let (rho2, d2) = (rho * rho % m, dm * dm % m);
        let x2 = (a2 + 2 * rho * dm + d2 * self.bk) % m;
        let x3 = (a3 + 3 * rho2 * dm + 3 * rho * d2 % m * self.bk + d2 * dm % m * self.b2k) % m;
        let narrow = |x: u64| u32::try_from(x).expect("below b^mid < 2^32");
        self.prefilter_digits(top, bmask, narrow(x2), narrow(x3))
    }

    /// The prefilter's digits of `n²` and `n³` (`x2`, `x3`, `mid` digits
    /// each, below `b^mid`) against each other, the bottom's digits and
    /// the certificate when it sits above them.
    #[inline]
    fn prefilter_digits(&self, top: &Top, bmask: u64, x2: u32, x3: u32) -> bool {
        fresh_digits(self.sb, self.mid, bmask | top.gmask, x2, x3)
    }
}

/// What a scan needs besides the masks: the top, the last bottom digit, the
/// residues and the scanned class's list entries.
struct Pass<'a> {
    top: &'a Top,
    d: u32,
    ext: &'a Ext,
    info: &'a [Info],
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::client_process::process_range_niceonly;
    use crate::overlap_join::join_range;
    use crate::overlap_join::test_fields::{
        FRONTIER_57, PLAN_FIELDS, WINDOWS, live_partitions, mid_mirror, reference_field,
    };
    use crate::stride_filter::StrideTable;
    use std::time::Instant;

    /// Dan Stoyell's windows, and base 10's whole band (which holds 69).
    fn windows() -> Vec<(u32, u128, u128, JoinParams)> {
        let mut v = vec![(10, 47, 100, JoinParams { t: 2, k: 1, p: 0 })];
        v.extend(
            WINDOWS
                .iter()
                .map(|&(b, s, e, t, k, p)| (b, s, e, JoinParams { t, k, p })),
        );
        v
    }

    /// Partition `v`'s tops, every key in order, as [`CpuJoin::run`] makes
    /// them.
    fn all_tops(join: &CpuJoin, v: u32) -> Vec<Top> {
        let (mut out, mut tops) = (Vec::new(), Vec::new());
        for key in 0..join.keyspace as usize {
            let (lo, hi) = (join.key_off[key] as usize, join.key_off[key + 1] as usize);
            join.tops(v, &join.fs.tlay[lo..hi], &mut tops);
            out.extend_from_slice(&tops);
        }
        out
    }

    /// The join of `range` with every top block certified by `Base::cert`.
    fn divided(b: u32, range: &FieldSize, jp: JoinParams) -> CpuJoin {
        CpuJoin {
            fd: None,
            ..CpuJoin::new(b, range, jp).expect("field setup")
        }
    }

    /// Up to `samples` of the field's live partitions, each against the
    /// reference join: the same join survivor count, the same prefilter
    /// survivors (by the prefilter's definition) and the same hits. Every
    /// key's tops are also those of [`divided`].
    fn check_partitions(b: u32, s: u128, e: u128, jp: JoinParams, samples: usize) {
        let join = CpuJoin::new(b, &FieldSize::new(s, e), jp).expect("field setup");
        let plain = divided(b, &FieldSize::new(s, e), jp);
        let live = live_partitions(&join.fs);
        let step = live.len().div_ceil(samples).max(1);
        let mut scratch = Scratch::default();
        for v in live.into_iter().step_by(step) {
            assert_eq!(
                all_tops(&join, v),
                all_tops(&plain, v),
                "b{b} [{s}, {e}) {jp:?} partition {v}: tops"
            );
            let mut cpu = Vec::new();
            let st = join_range(
                &join.fs.base,
                s,
                e,
                jp,
                Some(&[u128::from(v)]),
                Some(&mut cpu),
            );
            cpu.retain(|&n| mid_mirror(&join.fs, n));
            cpu.sort_unstable();
            let mut got = Vec::new();
            let res = join.run(v, &mut scratch, Some(&mut got));
            got.sort_unstable();
            let mut hits = res.hits.clone();
            hits.sort_unstable();
            assert_eq!(
                res.survivors, st.survivors,
                "b{b} [{s}, {e}) {jp:?} partition {v}: join survivors"
            );
            assert_eq!(
                got, cpu,
                "b{b} [{s}, {e}) {jp:?} partition {v}: prefilter survivors"
            );
            assert_eq!(
                res.checked,
                cpu.len() as u64,
                "b{b} [{s}, {e}) {jp:?} partition {v}"
            );
            assert_eq!(hits, st.hits, "b{b} [{s}, {e}) {jp:?} partition {v}: hits");
        }
    }

    /// The client's CPU stride path over `range` on `threads` threads: its
    /// nice numbers, sorted.
    fn stride_hits(b: u32, range: &FieldSize, threads: usize) -> Vec<u128> {
        use std::sync::atomic::{AtomicUsize, Ordering};
        // The client's CPU table (`DEFAULT_LSD_K_VALUE`).
        let table = StrideTable::new(b, 3);
        let chunks = range.chunks(range.size().div_ceil(64 * threads as u128));
        let next = AtomicUsize::new(0);
        let mut hits: Vec<u128> = std::thread::scope(|sc| {
            let workers: Vec<_> = (0..threads)
                .map(|_| {
                    sc.spawn(|| {
                        let mut hits = Vec::new();
                        while let Some(c) = chunks.get(next.fetch_add(1, Ordering::Relaxed)) {
                            let res = process_range_niceonly(c, b, &table);
                            hits.extend(res.nice_numbers.iter().map(|x| x.number));
                        }
                        hits
                    })
                })
                .collect();
            workers
                .into_iter()
                .flat_map(|w| w.join().expect("a stride worker panicked"))
                .collect()
        });
        hits.sort_unstable();
        hits
    }

    #[test]
    fn cpu_join_matches_the_reference_on_the_test_windows() {
        for (b, s, e, jp) in windows() {
            check_partitions(b, s, e, jp, 96);
        }
    }

    /// The stepped top certificates cover the test windows (all but base
    /// 10's, whose blocks are single numbers). Their blocks, full ones, are
    /// all seeded: `e² − a² >= (b^f0 − 1)·2b^(L−1)` has `L + f0 >= k + 2`
    /// digits, so the certificate sits above the prefilter's digits.
    #[test]
    fn the_test_windows_step_their_top_certificates() {
        for (b, s, e, jp) in windows() {
            let join = CpuJoin::new(b, &FieldSize::new(s, e), jp).expect("field setup");
            assert_eq!(join.fd.is_some(), b != 10, "b{b} [{s}, {e}) {jp:?}");
            if join.fd.is_some() {
                assert!(join.fs.full_floor >= join.fs.k2, "b{b} [{s}, {e}) {jp:?}");
            }
        }
    }

    /// Every top of sampled partitions of the production fields (the first
    /// and last live ones among them), every key: the stepped certificates
    /// give the tops [`divided`] gives.
    #[test]
    fn stepped_tops_equal_divided_tops_on_production_fields() {
        for &(b, s, size) in PLAN_FIELDS {
            let range = FieldSize::new(s, s + size);
            let jp = join_params_for(b, &range).expect("a join field");
            let join = CpuJoin::new(b, &range, jp).expect("field setup");
            assert!(join.fd.is_some(), "b{b}: no stepped certificates");
            let plain = divided(b, &range, jp);
            let live = live_partitions(&join.fs);
            let mut vs: Vec<u32> = live
                .iter()
                .copied()
                .step_by(live.len().div_ceil(24).max(1))
                .collect();
            if vs.last() != live.last() {
                vs.extend(live.last());
            }
            let mut tops = 0;
            for &v in &vs {
                let got = all_tops(&join, v);
                assert_eq!(got, all_tops(&plain, v), "b{b} partition {v}");
                tops += got.len();
            }
            eprintln!("b{b} {jp:?}: {} partitions, {tops} tops", vs.len());
        }
    }

    /// Every AND scan this CPU has (AVX-512F, AVX2, the portable loop)
    /// finds the same pairs: the same hits, join survivors and prefilter
    /// survivors on every partition of the test windows.
    #[test]
    fn every_scan_finds_the_same_pairs() {
        for (b, s, e, jp) in windows() {
            let range = FieldSize::new(s, e);
            let best = CpuJoin::new(b, &range, jp).expect("field setup");
            #[allow(unused_mut)]
            let mut lesser: Vec<CpuJoin> = Vec::new();
            #[cfg(target_arch = "x86_64")]
            for (avx512, avx2) in [(false, true), (false, false)] {
                let mut join = CpuJoin::new(b, &range, jp).expect("field setup");
                join.avx512 &= avx512;
                join.avx2 &= avx2;
                lesser.push(join);
            }
            let mut sc = Scratch::default();
            for v in 0..best.partitions() {
                let want = best.run_partition(v, &mut sc);
                for join in &lesser {
                    assert_eq!(
                        join.run_partition(v, &mut sc),
                        want,
                        "b{b} [{s}, {e}) {jp:?} partition {v}"
                    );
                }
            }
        }
    }

    /// Whole windows through [`CpuJoin::run_all`] find what the client's
    /// stride path finds.
    #[test]
    fn cpu_join_agrees_with_the_stride_path_on_whole_windows() {
        for (b, s, e, jp) in windows() {
            let range = FieldSize::new(s, e);
            let join = CpuJoin::new(b, &range, jp).expect("field setup");
            assert_eq!(
                join.run_all(2).hits,
                stride_hits(b, &range, 2),
                "b{b} [{s}, {e}) {jp:?}"
            );
        }
    }

    /// A field run slice by slice finds what it finds whole: the same hits,
    /// join survivors and prefilter survivors, on every window that has
    /// prefixes enough to cut into three slices.
    #[test]
    fn a_field_in_slices_finds_what_it_finds_whole() {
        let mut sliced_any = 0;
        for (b, s, e, jp) in windows() {
            let range = FieldSize::new(s, e);
            let whole = CpuJoin::new(b, &range, jp).expect("setup");
            let block = prefix_block(b, whole.fs.base.l, jp);
            let prefixes = range.last() / block - range.start() / block + 1;
            if prefixes < 3 {
                continue;
            }
            let slices = join_slices(&range, block, prefixes.div_ceil(3));
            let mut sc = Scratch::default();
            let run = |j: &CpuJoin, sc: &mut Scratch| {
                let mut out = PartitionResult::default();
                for v in 0..j.partitions() {
                    out.add(j.run_partition(v, sc));
                }
                out.hits.sort_unstable();
                out
            };
            let all = run(&whole, &mut sc);
            let mut parts = PartitionResult::default();
            for sl in &slices {
                parts.add(run(&CpuJoin::new(b, sl, jp).expect("slice setup"), &mut sc));
            }
            parts.hits.sort_unstable();
            assert_eq!(
                parts,
                all,
                "b{b} [{s}, {e}) {jp:?} in {} slices",
                slices.len()
            );
            sliced_any += 1;
        }
        assert!(sliced_any >= 5, "only {sliced_any} windows were cut");
    }

    /// The client's slices: one up to the cap, and the field's range in
    /// order beyond it.
    #[test]
    fn production_fields_are_one_slice_on_the_cpu() {
        for &(base, start, size) in PLAN_FIELDS {
            let (_, slices) =
                slices_for(base, &FieldSize::new(start, start + size)).expect("a join field");
            assert_eq!(slices.len(), 1, "b{base}");
        }
        assert!(
            slices_for(
                57,
                &FieldSize::new(FRONTIER_57, FRONTIER_57 + 4_000_000_000)
            )
            .is_none()
        );
    }

    /// Partition `v`'s bottom lists, every key digit in order (as
    /// [`CpuJoin::run`] makes them): `(key, class, bpre residue, mask,
    /// carries)` per listed bottom.
    fn listed(join: &CpuJoin, v: u32, ext: &mut Ext) -> Vec<(u32, usize, u32, u64, u32)> {
        let keys: Vec<u32> = if join.fs.key_level {
            (0..join.keyspace).collect()
        } else {
            vec![join.partition_digit(v, join.fs.jp.p - 1)]
        };
        let (mut masks, mut info, mut offs) = (Vec::new(), Vec::new(), Vec::new());
        join.extend(v, ext);
        let mut out = Vec::new();
        for d in keys {
            join.list(d, ext, &mut masks, &mut info, &mut offs);
            for c in 0..join.fs.m1 as usize {
                for i in offs[c] as usize..offs[c + 1] as usize {
                    let r0 = join.fs.bp_r[ext.idx[info[i].row as usize] as usize];
                    out.push((d, c, r0, masks[i], info[i].carries));
                }
            }
        }
        out
    }

    /// The field prune drops exactly the bottoms that meet the field's
    /// certificate: on every production field, `bpre` and the bottom lists
    /// of sampled partitions (every key digit) equal the unpruned ones
    /// filtered by it. Prints the counts before and after.
    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn the_field_prune_drops_exactly_the_bottoms_that_meet_the_certificate() {
        for &(b, s, size) in PLAN_FIELDS {
            let range = FieldSize::new(s, s + size);
            let jp = join_params_for(b, &range).expect("a join field");
            let join = CpuJoin::new(b, &range, jp).expect("field setup");
            let fc = join.fs.fc;
            assert_eq!(Some(fc), join.fs.base.cert(s, s + size - 1, jp.k));
            let mut raw = CpuJoin::new(b, &range, jp).expect("field setup");
            raw.fs.fc = 0;
            (raw.fs.bp_r, raw.fs.bp_m, raw.fs.seg) =
                FieldSetup::prune_bottoms(&raw.fs.bp_all, raw.fs.m1, 0).expect("bpre");
            assert_eq!(raw.fs.bp_r, raw.fs.bp_all.0, "b{b}: fc = 0 prunes nothing");
            let every = u64::MAX >> (64 - b);
            assert_eq!(
                FieldSetup::prune_bottoms(&raw.fs.bp_all, raw.fs.m1, every).expect("bpre"),
                (raw.fs.bp_r.clone(), raw.fs.bp_m.clone(), raw.fs.seg.clone()),
                "b{b}: a certificate that meets every residue prunes nothing"
            );
            let class = |seg: &[u32], i: usize| seg.partition_point(|&x| x as usize <= i) - 1;
            let pre = |fs: &FieldSetup| -> Vec<(usize, u32, u64)> {
                (0..fs.bp_r.len())
                    .map(|i| (class(&fs.seg, i), fs.bp_r[i], fs.bp_m[i]))
                    .collect()
            };
            let mut want = pre(&raw.fs);
            want.retain(|&(_, _, m)| m & fc == 0);
            assert_eq!(pre(&join.fs), want, "b{b}: bpre");
            let live = live_partitions(&join.fs);
            let (mut ext, mut ext_raw) = (Ext::default(), Ext::default());
            let (mut before, mut after, mut nparts) = (0, 0, 0);
            for v in live.iter().copied().step_by(live.len().div_ceil(2)) {
                let got = listed(&join, v, &mut ext);
                let mut want = listed(&raw, v, &mut ext_raw);
                before += want.len();
                want.retain(|&(_, _, _, m, _)| m & fc == 0);
                assert_eq!(got, want, "b{b} partition {v}: bottom lists");
                after += got.len();
                nparts += 1;
            }
            eprintln!(
                "b{b} [{s}, +{size}) {jp:?}: field certificate {} digits; bpre {} -> {}; \
                 listed bottoms per partition (all keys) {} -> {} ({:.1}% pruned)",
                fc.count_ones(),
                raw.fs.bp_r.len(),
                join.fs.bp_r.len(),
                before / nparts,
                after / nparts,
                100.0 * (1.0 - after as f64 / before as f64)
            );
        }
    }

    /// splitmix64, for the random tests.
    fn next_u64(x: &mut u64) -> u64 {
        *x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *x;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A random number below `m` (multiply-shift).
    #[allow(clippy::cast_possible_truncation)]
    fn below(x: &mut u64, m: u32) -> u32 {
        (((next_u64(x) >> 32) * u64::from(m)) >> 32) as u32
    }

    /// A random set of exactly `size` of the digits below `b`.
    fn random_set(x: &mut u64, b: u32, size: u32) -> u64 {
        let mut digits: [u32; 64] = std::array::from_fn(|i| u32::try_from(i).expect("< 64"));
        let mut set = 0;
        for i in 0..size {
            let pick = (i + below(x, b - i)) as usize;
            digits.swap(i as usize, pick);
            set |= 1u64 << digits[i as usize];
        }
        set
    }

    /// The fixed-point digits ([`Fx`]) at every base 10–64: the constants
    /// and `δ̂` against exact ceilings for every `D < b²`, the error bound
    /// `Δ < 2^32`, and the digits against `(A + P·D) mod b²` for every `(P,
    /// D)` at the largest `A` and every `(A, D)` at the largest `P`.
    #[test]
    fn fx_digits_match_the_division() {
        let two32 = 1u64 << 32;
        let mut n = 0u64;
        for b in 10..=64u32 {
            let fx = Fx::new(b);
            let (b2, bb) = (b * b, u64::from(b));
            let b2w = u64::from(b2);
            assert_eq!(u64::from(fx.c), two32.div_ceil(b2w), "b{b}: C");
            assert_eq!(u64::from(fx.q) * b2w + u64::from(fx.r), two32, "b{b}: q, r");
            assert!(u64::from(fx.r) < b2w, "b{b}: r");
            let e1 = u64::from(fx.c) * b2w - two32;
            let dhs: Vec<u32> = (0..b2).map(|d| fx.dh(d)).collect();
            let mut e2 = 0;
            for (d, &dh) in (0u64..).zip(&dhs) {
                assert_eq!(u64::from(dh), (d << 32).div_ceil(b2w), "b{b} D {d}: δ̂");
                e2 = e2.max(u64::from(dh) * b2w - (d << 32));
            }
            assert!(e1 < b2w && e2 < b2w, "b{b}: ε");
            assert!((b2w - 1) * (e1 + e2) < two32, "b{b}: Δ");
            let top = b2 - 1;
            for x in 0..b2 {
                for (d, &dh) in (0..b2).zip(&dhs) {
                    for (a, p) in [(top, x), (x, top)] {
                        let want = (u64::from(a) + u64::from(p) * u64::from(d)) % b2w;
                        let (h, g) = fx.digits(a, p, dh);
                        assert!(g < bb && h * bb + g == want, "b{b} A {a} P {p} D {d}");
                    }
                }
            }
            n += 2 * b2w * b2w;
        }
        eprintln!("FX: {n} (A, P, D) at bases 10-64 equal the division");
    }

    /// Opt-in (`NICE_TEST_FX_EXHAUSTIVE=1`; minutes on every core): the
    /// fixed-point digits for every `A, P, D < b²` at every base 10–64.
    #[test]
    #[ignore = "opt-in, minutes of CPU"]
    fn fx_digits_are_exact_everywhere() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        if std::env::var("NICE_TEST_FX_EXHAUSTIVE").is_err() {
            eprintln!("skipping: set NICE_TEST_FX_EXHAUSTIVE to check every (A, P, D)");
            return;
        }
        let items: Vec<(u32, u32)> = (10..=64u32)
            .rev()
            .flat_map(|b| (0..b * b).map(move |d| (b, d)))
            .collect();
        let threads = std::thread::available_parallelism().map_or(4, usize::from);
        let next = AtomicUsize::new(0);
        let t = Instant::now();
        let n: u64 = std::thread::scope(|s| {
            let workers: Vec<_> = (0..threads)
                .map(|_| {
                    s.spawn(|| {
                        let mut n = 0u64;
                        while let Some(&(b, d)) = items.get(next.fetch_add(1, Ordering::Relaxed)) {
                            let fx = Fx::new(b);
                            let (b2, bb, dh) = (b * b, u64::from(b), fx.dh(d));
                            for p in 0..b2 {
                                // x = (A + P·D) mod b² for A = 0, 1, ...
                                let x0 = p * d % b2;
                                let split = b2 - x0;
                                let mut bad = 0u32;
                                for a in 0..b2 {
                                    let want = x0 + a - if a >= split { b2 } else { 0 };
                                    let (h, g) = fx.digits(a, p, dh);
                                    bad += u32::from(h * bb + g != u64::from(want));
                                }
                                assert_eq!(bad, 0, "b{b} P {p} D {d}");
                                n += u64::from(b2);
                            }
                        }
                        n
                    })
                })
                .collect();
            workers
                .into_iter()
                .map(|w| w.join().expect("an FX worker panicked"))
                .sum()
        });
        eprintln!(
            "FX exhaustive: {n} (A, P, D) at bases 10-64 on {threads} threads in {:.0}s",
            t.elapsed().as_secs_f64()
        );
    }

    /// The shift membership ([`fresh4`]) is the four one-hot masks'
    /// predicate ([`fresh_digits`] at `mid = 2`): every digit 4-tuple at
    /// every base 10–64 against two random sets, and 200,000 random tuples
    /// per base against sets of every size.
    #[test]
    fn lean_membership_matches_the_masks() {
        let mut x = 0x2545_F491_4F6C_DD1D;
        for b in 10..=64u32 {
            let sb = SmallDiv::new(u64::from(b));
            let sets = [random_set(&mut x, b, b / 4), random_set(&mut x, b, b / 2)];
            for seen in sets {
                for h2 in 0..b {
                    for g2 in 0..b {
                        for h3 in 0..b {
                            for g3 in 0..b {
                                let want = fresh_digits(sb, 2, seen, h2 * b + g2, h3 * b + g3);
                                let got = fresh4(seen, h2.into(), g2.into(), h3.into(), g3.into());
                                assert_eq!(
                                    got, want,
                                    "b{b} {seen:#x} h2 {h2} g2 {g2} h3 {h3} g3 {g3}"
                                );
                            }
                        }
                    }
                }
            }
            for i in 0..200_000u32 {
                let seen = random_set(&mut x, b, i % (b + 1));
                let [h2, g2, h3, g3] = std::array::from_fn(|_| below(&mut x, b));
                let want = fresh_digits(sb, 2, seen, h2 * b + g2, h3 * b + g3);
                let got = fresh4(seen, h2.into(), g2.into(), h3.into(), g3.into());
                assert_eq!(got, want, "b{b} {seen:#x} h2 {h2} g2 {g2} h3 {h3} g3 {g3}");
            }
        }
    }

    /// The fixed-point prefilter ([`Fx::pass`]) against the division form
    /// (the digits of `(A + P·D) mod b²` by [`fresh_digits`]) on 10^6
    /// random pairs per base 10–64: carries, coefficients and `D` below
    /// `b²`, and sets of every size from 0 to `b`.
    #[test]
    fn lean_prefilter_matches_the_division_form() {
        let mut x = 0x6A09_E667_F3BC_C908;
        for b in 10..=64u32 {
            let (fx, sb, b2) = (Fx::new(b), SmallDiv::new(u64::from(b)), b * b);
            let mut passed = 0u32;
            for i in 0..1_000_000u32 {
                let seen = random_set(&mut x, b, i % (b + 1));
                let [a2, a3, p2, p3, d] = std::array::from_fn(|_| below(&mut x, b2));
                let want = fresh_digits(sb, 2, seen, (a2 + p2 * d) % b2, (a3 + p3 * d) % b2);
                let got = fx.pass(seen, a2 | a3 << 16, p2, p3, fx.dh(d));
                assert_eq!(got, want, "b{b} {seen:#x} A {a2} {a3} P {p2} {p3} D {d}");
                passed += u32::from(want);
            }
            assert!(passed > 0 && passed < 1_000_000, "b{b}: {passed} passed");
        }
    }

    /// The fixed-point prefilter covers exactly the shapes with `mid = 2`
    /// and `k > 2`: every production field, and every test window but
    /// b10's `k = 1` and b20's `k = 2`, which take the general form.
    #[test]
    fn production_shapes_take_the_fixed_point_prefilter() {
        for &(b, s, size) in PLAN_FIELDS {
            let range = FieldSize::new(s, s + size);
            let jp = join_params_for(b, &range).expect("a join field");
            let join = CpuJoin::new(b, &range, jp).expect("field setup");
            assert!(join.lean && join.mid == 2, "b{b} {jp:?}");
        }
        for (b, s, e, jp) in windows() {
            let join = CpuJoin::new(b, &FieldSize::new(s, e), jp).expect("field setup");
            assert_eq!(
                join.lean,
                join.mid == 2 && jp.k > 2,
                "b{b} [{s}, {e}) {jp:?}"
            );
        }
    }

    #[test]
    fn cpu_join_finds_69_in_base_10() {
        let join = CpuJoin::new(
            10,
            &FieldSize::new(47, 100),
            JoinParams { t: 2, k: 1, p: 0 },
        )
        .expect("field setup");
        assert_eq!(join.run_all(2).hits, vec![69]);
    }

    /// Opt-in (`NICE_TEST_CPU_JOIN_REF=n`; minutes): `n` partitions of each
    /// production-size field, with the production parameters, against the
    /// reference join.
    #[test]
    #[ignore = "opt-in, minutes"]
    fn cpu_join_matches_the_reference_on_production_fields() {
        let Ok(n) = std::env::var("NICE_TEST_CPU_JOIN_REF") else {
            eprintln!("skipping: set NICE_TEST_CPU_JOIN_REF to a partition count per field");
            return;
        };
        let n: usize = n
            .parse()
            .expect("NICE_TEST_CPU_JOIN_REF: a partition count");
        for &(b, s, size) in PLAN_FIELDS {
            let range = FieldSize::new(s, s + size);
            let jp = join_params_for(b, &range).expect("a join field");
            let t = Instant::now();
            check_partitions(b, s, s + size, jp, n);
            eprintln!("b{b} {jp:?}: {:.1}s", t.elapsed().as_secs_f64());
        }
    }

    /// Opt-in (`NICE_TEST_JOIN_FULL_FIELD=1`, as for the GPU; minutes of
    /// CPU): the whole base-42 field of the GPU's check through
    /// [`CpuJoin::run_all`] on every core, against the reference over every
    /// partition (the same join survivors, prefilter survivors and hits)
    /// and the client's stride path (the same nice numbers).
    #[test]
    #[ignore = "opt-in, minutes of CPU"]
    fn cpu_join_matches_the_reference_on_a_whole_field() {
        if std::env::var("NICE_TEST_JOIN_FULL_FIELD").is_err() {
            eprintln!("skipping: set NICE_TEST_JOIN_FULL_FIELD to run the whole-field comparison");
            return;
        }
        let (base, start, size) = PLAN_FIELDS[0];
        let range = FieldSize::new(start, start + size);
        let join = CpuJoin::for_field(base, &range).expect("a join field");
        let threads = std::thread::available_parallelism().map_or(4, usize::from);
        let t = Instant::now();
        let (survivors, checked, hits) = reference_field(&join.fs);
        let ref_secs = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let got = join.run_all(threads);
        let join_secs = t.elapsed().as_secs_f64();
        assert_eq!(
            (got.survivors, got.checked, &got.hits),
            (survivors, checked, &hits),
            "b{base} {range:?}: the join differs from the reference"
        );
        let t = Instant::now();
        assert_eq!(
            stride_hits(base, &range, threads),
            hits,
            "b{base} {range:?}: the stride path differs"
        );
        eprintln!(
            "FULL FIELD cpu b{base} {range:?}: {} partitions, {survivors} join survivors, \
             {checked} checked, hits {hits:?}; on {threads} threads: reference {ref_secs:.0}s, \
             join {join_secs:.1}s, stride {:.1}s",
            join.partitions(),
            t.elapsed().as_secs_f64()
        );
    }

    /// Opt-in timing (`NICE_TEST_CPU_JOIN=n`): every `n`th partition of a
    /// base-57 frontier field on one thread, and the client's stride path
    /// on random chunks of the same field (`NICE_TEST_CPU_JOIN_CHUNKS`,
    /// default 120, of the client's 1e9: their cost is heavy-tailed), each
    /// extrapolated to the whole field on one thread.
    #[test]
    #[ignore = "opt-in timing"]
    #[allow(clippy::cast_precision_loss)]
    fn cpu_join_partition_timing() {
        let Ok(step) = std::env::var("NICE_TEST_CPU_JOIN") else {
            eprintln!("skipping: set NICE_TEST_CPU_JOIN to a partition stride");
            return;
        };
        let step: u32 = step.parse().expect("NICE_TEST_CPU_JOIN: a stride");
        let chunks: u32 = std::env::var("NICE_TEST_CPU_JOIN_CHUNKS").map_or(120, |s| {
            s.parse().expect("NICE_TEST_CPU_JOIN_CHUNKS: a count")
        });
        let (size, chunk) = (100_000_000_000_000u128, 1_000_000_000u128);
        let range = FieldSize::new(FRONTIER_57, FRONTIER_57 + size);
        let join = CpuJoin::for_field(57, &range).expect("a join field");
        let mut scratch = Scratch::default();
        let t = Instant::now();
        let (mut n, mut total) = (0u32, PartitionResult::default());
        for v in (0..join.partitions()).step_by(step as usize) {
            total.add(join.run_partition(v, &mut scratch));
            n += 1;
        }
        let secs = t.elapsed().as_secs_f64();
        let per = secs / f64::from(n);
        let join_field = per * f64::from(join.partitions()) + join.setup_secs();
        let table = StrideTable::new(57, 3);
        let mut x = 0x9E37_79B9_7F4A_7C15u64;
        let t = Instant::now();
        for _ in 0..chunks {
            x = x
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let s = FRONTIER_57 + u128::from(x >> 33) % (size / chunk) * chunk;
            let _ = process_range_niceonly(&FieldSize::new(s, s + chunk), 57, &table);
        }
        let stride_field = t.elapsed().as_secs_f64() / f64::from(chunks) * (size / chunk) as f64;
        eprintln!(
            "CPU JOIN b57 frontier: setup {:.3}s, {n} partitions in {secs:.2}s ({:.1} ms each), \
             field ~{join_field:.0}s on one thread; survivors {} checked {} hits {}; stride \
             path ~{stride_field:.0}s on one thread ({:.1}x)",
            join.setup_secs(),
            per * 1e3,
            total.survivors,
            total.checked,
            total.hits.len(),
            stride_field / join_field
        );
    }
}
