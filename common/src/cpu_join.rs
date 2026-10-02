//! The overlap join on the CPU. [`crate::overlap_join`] explains the join
//! and holds the reference it is tested against; `crate::cubecl_join` is
//! the same join on a GPU.
//!
//! A field is `b^p` partitions, independent of each other, which the client
//! runs in parallel ([`CpuJoin::run_partition`]). One partition:
//!
//! 1. **Tops**: every top-layer prefix extended by the partition's digits
//!    and certified, as in the reference join.
//! 2. **Bottoms**: every residue of the field's bottom list extended by the
//!    partition's fixed digits, with exact arithmetic once per residue.
//!    What the last bottom digit `d` (the key digit) adds is affine in `d`:
//!    the output digits at position `k − 1`, and the prefilter's above it,
//!    come from `(q + d·c) mod b^(1 + mid)`.
//! 3. **Join**, one key digit at a time: the key's bottoms are listed by
//!    digit-sum class (about a megabyte at base 57, so they stay in cache),
//!    and each of the key's tops scans the classes its roots name, one
//!    64-bit AND per bottom.
//! 4. **Survivors** go through the middle-digit prefilter, as on the GPU,
//!    and then the client's own full check.
//!
//! The client sends it the nice-only fields the GPU's join would take
//! ([`CpuJoin::for_field`]). Its survivors, and the prefilter's, are exactly
//! the reference join's: the tests compare them partition by partition.

use crate::FieldSize;
use crate::client_process::{get_is_nice, get_is_nice_with_known_lsd};
use crate::overlap_join::{Div64, FieldSetup, JoinParams, join_params_for};
use anyhow::Result;

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
    fn add(&mut self, other: PartitionResult) {
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
struct Top {
    p: u128,
    /// The certificate: output digits at positions `>= k`.
    mask: u64,
    rlo: u32,
    rhi: u32,
    /// `n`'s digits `k..k2` (the next digits of `P`), as a number.
    dm: u32,
    /// Every certified position is `>= k2`, so the prefilter's digits must
    /// avoid the certificate too.
    seed: bool,
    /// `P mod (b − 1)`, and the key digit.
    pc: u32,
    key: u32,
}

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
struct SmallDiv {
    d: u64,
    magic: u64,
    magic32: u64,
}

// The quotients and remainders are below the bounds the methods state, and
// `d` below 2^32 wherever it is narrowed.
#[allow(clippy::cast_possible_truncation)]
impl SmallDiv {
    fn new(d: u64) -> Self {
        Self {
            d,
            magic: (1u64 << 48).div_ceil(d),
            magic32: (1u64 << 32).div_ceil(d),
        }
    }

    /// `(x / d, x % d)`.
    #[inline]
    fn divrem(self, x: u64) -> (u64, u64) {
        let q = ((u128::from(x) * u128::from(self.magic)) >> 48) as u64;
        (q, x - q * self.d)
    }

    /// `x % d` for `x < 2^48/d` and `d < 2^32`.
    #[inline]
    fn rem(self, x: u64) -> u32 {
        self.divrem(x).1 as u32
    }

    /// `(x / d, x % d)` with one 64-bit multiply, exact below `2^32/d`
    /// (which `2^26` is for `d <= 64`).
    #[inline]
    fn divrem_small(self, x: u32) -> (u32, u32) {
        let q = ((u64::from(x) * self.magic32) >> 32) as u32;
        (q, x - q * self.d as u32)
    }
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
    smid: SmallDiv,
    /// The AND scan may use AVX2 (detected at run time).
    #[cfg(target_arch = "x86_64")]
    avx2: bool,
}

impl CpuJoin {
    /// The CPU join for a nice-only field, if it takes the field: the same
    /// gate as the GPU's (`overlap_join::join_params_for`).
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
        let fs = FieldSetup::new(base, range.start(), range.end(), jp)?;
        let b = u64::from(base);
        let k = jp.k;
        let mid = fs.k2 - k;
        let big_m = b.pow(1 + mid);
        let bmid = b.pow(mid);
        let bk1 = b.pow(k - 1);
        Ok(Self {
            k,
            mid,
            keyspace: if fs.key_level { base } else { 1 },
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
            smid: SmallDiv::new(bmid),
            #[cfg(target_arch = "x86_64")]
            avx2: std::arch::is_x86_feature_detected!("avx2"),
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
        self.tops(v, &mut sc.tops);
        if sc.tops.is_empty() {
            return out;
        }
        self.extend(v, &mut sc.ext);
        sc.tops.sort_unstable_by_key(|t| (t.key, t.pc));
        let m1 = self.fs.m1;
        let roots = &self.fs.base.roots;
        let mut first = 0;
        while first < sc.tops.len() {
            let key = sc.tops[first].key;
            let end = first + sc.tops[first..].partition_point(|t| t.key == key);
            // The last bottom digit: the key itself, or the partition's last.
            let d = if self.fs.key_level {
                key
            } else {
                self.partition_digit(v, self.fs.jp.p - 1)
            };
            self.list(d, &mut sc.ext, &mut sc.masks, &mut sc.info, &mut sc.offs);
            for top in &sc.tops[first..end] {
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
            first = end;
        }
        out
    }

    /// The partition's tops, as the reference join makes them.
    // Below 2^32: w = b^f0 is (`JoinParams::supported`), and so are b^mid,
    // b - 1 and the key space. w, p, a, e as in `overlap_join`.
    #[allow(clippy::cast_possible_truncation, clippy::many_single_char_names)]
    fn tops(&self, v: u32, tops: &mut Vec<Top>) {
        tops.clear();
        let fs = &self.fs;
        let (w, v) = (fs.w, u128::from(v));
        for &(p0, _) in &fs.tlay {
            let p = p0 * fs.nparts + v;
            if p < fs.plo || p > fs.phi {
                continue;
            }
            let lo = p * w;
            let a = lo.max(fs.s);
            let e = (lo + w - 1).min(fs.e - 1);
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
                rlo: (a - lo) as u32,
                rhi: (e - lo + 1) as u32,
                dm: ((p / self.pk) % u128::from(self.bmid)) as u32,
                seed: floor >= fs.k2,
                pc: (p % u128::from(fs.m1)) as u32,
                key: (p0 % u128::from(self.keyspace)) as u32,
            });
        }
    }

    /// The bottom list extended by the partition's fixed digits: the
    /// positions from `f0` up to `k − 2` (`k − 1` holds the key digit, or
    /// the partition's last digit, which [`Self::list`] adds). Residues
    /// whose digits repeat are dropped here, once for every key.
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
                    if g2 == g3 || mask & bits != 0 {
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
        for c in 0..self.fs.m1 as usize {
            offs.push(masks.len() as u32);
            for row in ext.seg[c]..ext.seg[c + 1] {
                let i = row as usize;
                let (a2, g2) = self.sb.divrem_small(ext.t2[i]);
                let (a3, g3) = self.sb.divrem_small(ext.t3[i]);
                let bits = (1u64 << g2) | (1u64 << g3);
                let mask = ext.mask[i];
                if g2 == g3 || mask & bits != 0 {
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
        if self.avx2 {
            // SAFETY: `avx2` is only set where the CPU has AVX2.
            unsafe { self.scan_avx2(pass, masks, out, record) };
            return;
        }
        self.scan_any(pass, masks, out, record);
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
        let pass_mid = if self.kb == 0 && self.bk == 0 && self.b2k == 0 {
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
    /// `⌊R³/b^k⌋ + 3R²D` mod `b^mid` when `k >= mid` (the production
    /// parameters), with `2R`, `3R²` (`p2`, `p3`) the bottom's.
    #[inline]
    fn prefilter(&self, top: &Top, bmask: u64, carries: u32, p2: u32, p3: u32) -> bool {
        let dm = u64::from(top.dm);
        let x2 = self
            .smid
            .rem(u64::from(carries & 0xFFFF) + u64::from(p2) * dm);
        let x3 = self.smid.rem(u64::from(carries >> 16) + u64::from(p3) * dm);
        self.prefilter_digits(top, bmask, x2, x3)
    }

    /// [`Self::prefilter`] for any parameters: also the terms in `b^k`
    /// (`D²b^k`, `3RD²b^k`, `D³b^(2k)`), which vanish unless `k` is below
    /// the prefilter's depth, and `R mod b^mid` as given.
    fn prefilter_any(&self, top: &Top, bmask: u64, carries: u32, rho: u64) -> bool {
        let (m, dm) = (self.bmid, u64::from(top.dm));
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
    fn prefilter_digits(&self, top: &Top, bmask: u64, mut x2: u32, mut x3: u32) -> bool {
        let mut seen = bmask | if top.seed { top.mask } else { 0 };
        for _ in 0..self.mid {
            let (q2, g2) = self.sb.divrem_small(x2);
            let (q3, g3) = self.sb.divrem_small(x3);
            let bits = (1u64 << g2) | (1u64 << g3);
            if g2 == g3 || seen & bits != 0 {
                return false;
            }
            seen |= bits;
            (x2, x3) = (q2, q3);
        }
        true
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

    /// Up to `samples` of the field's live partitions, each against the
    /// reference join: the same join survivor count, the same prefilter
    /// survivors (by the prefilter's definition) and the same hits.
    fn check_partitions(b: u32, s: u128, e: u128, jp: JoinParams, samples: usize) {
        let join = CpuJoin::new(b, &FieldSize::new(s, e), jp).expect("field setup");
        let live = live_partitions(&join.fs);
        let step = live.len().div_ceil(samples).max(1);
        let mut scratch = Scratch::default();
        for v in live.into_iter().step_by(step) {
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
