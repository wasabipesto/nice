# Lean proofs for the nice-number search

Machine-checked statements of the mathematics the search relies on: the
definition of a nice number, the search interval per base, and the
soundness of every filter in the niceonly cascade, plus the structural and
negative results the project has accumulated. `DESIGN.md` has the plan and
phases; `CLAIMS.md` is the catalogue. The first Lean attempt (the `proofs`
branch, Dec 2025) supplied the definitions; the rest is new.

## Layout

```
Nice/Spec/      the mathematics: digits, IsNice, base range
Nice/Model/     executable mirrors of the Rust filters, proved sound against Spec
Nice/Theory/    structural facts and negative results independent of the code
Nice/Const.lean numeric constants the Rust relies on, certified
Nice/Refuted.lean, Nice/Conjectures.lean, Nice/Examples.lean
DESIGN.md       design, layers, phases
CLAIMS.md       the registry: claim id → Lean declaration → Rust site → status
scripts/check_claims.py   validates CLAIMS.md against the build and the Rust tags
fixtures/       tables emitted by Rust (`scripts/lean_fixtures.rs`), checked
                against the Lean model by `lake exe conformance`
Conformance.lean          the conformance executable
```

Three layers. **Spec** states mathematics only. **Model** contains
computable Lean functions that do what the Rust does at the algorithmic
level, each with a soundness theorem against Spec ("if the model rejects,
no nice number is lost"). **Theory** is what is true about the problem
independent of the code. The end-to-end targets are `END-1` and `END-2`:
the modelled stride pipeline and the modelled overlap join each report
exactly the nice numbers of a field.

Declarations live in the namespace `NiceSearch`; the library, its modules
and their paths stay `Nice` (`import Nice.Model.Cross`). The namespace
keeps clear of `Nice`, the root namespace of Janzert's
`nice-numbers-lean`, so either project can import the other.

## Building

```
elan toolchain install $(cat lean-toolchain)   # once
cd proofs
lake exe cache get      # Mathlib build cache, ~6 GB on disk, minutes
lake build
```

or `just lean-build` from the repo root. Mathlib is pinned to the release
tag matching `lean-toolchain`; bump both together, deliberately.

Policy: no `native_decide`; `decide` / `norm_num` for concrete examples;
milestone theorems must depend on no axioms beyond `propext`,
`Classical.choice`, `Quot.sound` (the checker enforces this for every row
marked `proved`).

## Fixtures

`just lean-fixtures` runs `scripts/lean_fixtures.rs` (rust-script), which
dumps the residue tables, LSD bitmaps, stride tables (residues, gaps,
low-digit masks, `first_valid_at_or_after` samples), base ranges and
seeded-check verdicts for small parameters into `fixtures/*.json`.
`just lean-conformance` recomputes each from the executable Lean model
and diffs. The fixtures also hold the affine filter's verdicts and the
overlap join on small windows: certificates, the bottom list, every
partition's survivors, the prefilter's count and the slicing. Theorems
pin "model = spec"; this pins "model = code". The fixtures are checked
in; regenerate them when the Rust tables change.

## The registry and the Rust tags

`CLAIMS.md` has one row per claim. A Rust doc comment of the form

```rust
/// Lean: `NiceSearch.mem_residueFilter_of_isNice` (RES-1)
```

ties a code site to a row. `just lean-claims` (after `lake build`) checks
that every tag names its row's declaration, that every `proved` declaration
exists and is sorry-free, and reports `stated`/`planned` declarations that
have become sorry-free so their status can be promoted. Proof debt is
allowed and visible: a filter PR may add a row and a tag whose theorem is
only `stated`. It may not silently un-prove something.

## Adding a claim (code → Lean)

1. Add a row to `CLAIMS.md` with the statement and explicit hypotheses.
   If the statement cannot be written down, that is the review finding.
2. Add the model function under `Nice/Model/` and, when tables are
   involved, a fixture the Rust emits for it.
3. State the soundness theorem; prove it, or mark the row `stated`.
4. Tag the Rust site.

## Proposing an optimization (Lean → code)

State it in `Nice/Conjectures.lean` as a soundness theorem with `sorry`
and a `decide` check on bases 5–16. A failing check moves it to
`Nice/Refuted.lean` with its witness. A passing one earns proof effort, and
the theorem's hypotheses are the implementation's spec.

## Status

<!-- status:begin -->
| phase | def | proved | stated | planned | other |
|---|---|---|---|---|---|
| 0 | 1 | 3 | 0 | 0 | 0 |
| 1 | 1 | 17 | 0 | 0 | 0 |
| 2 | 0 | 10 | 0 | 0 | 1 |
| 3 | 0 | 9 | 0 | 1 | 0 |
| 4 | 0 | 5 | 0 | 0 | 1 |
| 5 | 0 | 12 | 0 | 0 | 0 |
| 6 | 1 | 5 | 0 | 3 | 0 |
| 7 | 0 | 7 | 0 | 0 | 0 |
| — | 0 | 0 | 0 | 0 | 2 |

Proved or defined so far:

- **DEF-1** `NiceSearch.IsNice`: `IsNice b n` ⇔ the base-b digits of n² followed by those of n³ permute `0..b-1`
- **DEF-1a** `NiceSearch.isNice_iff_pandigital`: the three-part `Pandigital` definition of `origin/proofs` is equivalent
- **DEF-2** `NiceSearch.numUniques_eq_iff_isNice`: inside the base range, `numUniques b n = b ↔ IsNice b n`
- **DEF-3** `NiceSearch.one_le_numUniques`: `1 ≤ numUniques b n` for `n ≥ 1` (histogram bin 0 is empty)
- **DEF-4** `NiceSearch.nearMissCutoff`: near-miss cutoff is `⌊0.9 b⌋` with strict `>` (`IsNearMiss`); every nice number is a near miss (`isNearMiss_of_isNice`)
- **RNG-1** `NiceSearch.nice_digit_count`: `IsNice b n → numDigits(n²) + numDigits(n³) = b`
- **RNG-2** `NiceSearch.memBaseRange_of_inBaseRange`: the per-`b mod 5` closed-form interval contains every n with `numDigits(n²) + numDigits(n³) = b`
- **RNG-2b** `NiceSearch.inBaseRange_of_memBaseRange`: conversely every n in the closed-form interval has digit-count sum b; with RNG-2 the interval is exactly the search range (`memBaseRange_iff`)
- **RNG-3** `NiceSearch.three_le_numDigits_of_inBaseRange`: inside the range every power has `≥ k` digits for `k ≤ 3`, `b ≥ 6`
- **RNG-4** `NiceSearch.not_inBaseRange_of_one_mod_five`: `b ≡ 1 (mod 5)` ⇒ no n has digit-count sum b
- **RNG-5** `NiceSearch.numDigits_pow_mono`: `numDigits b (n^e)` is monotone in n
- **NUM-1** `NiceSearch.Const.u128_cutoff_40`: `(rangeEnd 40 − 1)^3 < 2^128`
- **NUM-2** `NiceSearch.Const.u256_cutoff`: `(rangeEnd b − 1)^3 < 2^256` for `b ≤ 68`; 69 fits, 70 does not
- **NUM-3** `NiceSearch.Const.max_fw_digits`: `numDigits b (n^3) ≤ 38` for `b ≤ 64` in range
- **NUM-4** `NiceSearch.Const.stride_modulus_u32`: `(b−1)·b^3 < 2^32` for `b ≤ 256` (u32 stride table)
- **NUM-4a** `NiceSearch.Const.stride_modulus_gpu`: `(b−1)·b^3 < 2^28` for `b ≤ 128` (`MAX_STRIDE_MODULUS`)
- **NUM-5** `NiceSearch.Const.mask_width`: digit masks need `b ≤ 64` (u64) / `b ≤ 128` (two words)
- **NUM-6** `NiceSearch.chunkExp_spec`: chunk constants: `b^e < bound ≤ b^(e+1)` for `e = chunkExp b bound` (maximal), for the GPU chunks (bounds `2^31`, `2^16`) and the MSD endpoint extraction's chunks (`chunk_digits`, bound `2^32`); and the split16 shift bound `(div−1)·2^16 < 2^32` (`split16_shift_bound`)
- **NUM-7** `NiceSearch.prefilter_sound`: the prefilter depth `min(numDigits(start²), numDigits(start³)) − 1` computed exactly from the range start is sound for every candidate at or after it
- **NUM-8** `NiceSearch.Const.mod_m_bound`: `M² + M < 2^64` for `M < 2^32`
- **NUM-9** `NiceSearch.Const.histogram_bins`: histogram bins cannot overflow u32
- **RES-1** `NiceSearch.mem_residueFilter_of_isNice`: `IsNice b n → n² + n³ ≡ b(b−1)/2 (mod b−1)`; `n mod (b−1) ∈ residueFilter b`
- **RES-1a** `NiceSearch.nice_digit_sum`: a nice number's output digits sum to `b(b−1)/2`
- **RES-2** `NiceSearch.no_nice_of_three_mod_four`: `b ≡ 3 (mod 4) → ∀ n, ¬IsNice b n`
- **RES-3** `NiceSearch.no_nice_of_residueFilter_empty`: `residueFilter b = ∅ → ∀ n, ¬IsNice b n`; `residueFilter 11 = ∅`
- **LSD-1** `NiceSearch.digit_pow_mod_pow`: `digit b (n^e) j` for `j < k` depends only on `n mod b^k`
- **LSD-2** `NiceSearch.mem_lsdBitmap_of_isNice`: nice + RNG-3 ⇒ the 2k fixed-width low digits are pairwise distinct ⇒ `n mod b^k ∈ lsdBitmap b k`
- **STR-1** `NiceSearch.mem_validResidues_iff`: `Coprime (b−1) (b^k)`; passes both ⇔ `n mod M ∈ validResidues`
- **STR-2** `NiceSearch.walk_eq_filter`: the gap-table walk visits exactly the valid n in `[start,end)` in order
- **STR-3** `NiceSearch.seeded_iff_isNice`: seeded check equals the plain check under RNG-3
- **STR-4** `NiceSearch.lowMask_eq`: `low_digit_masks[i]` is exactly the low-digit set of residue i's powers
- **GPU-0** `NiceSearch.exists_ordinal_eq`: ordinal formula `B0 + ⌊g/#V⌋·M + V[g mod #V]` is strictly increasing (`ordinal_strictMono`), always valid (`ordinal_valid`) and hits every valid n ≥ B0, so it enumerates the same set as STR-2
- **MSD-1** `NiceSearch.digit_mem_cyclicInterval`: interval digit domains are a superset of the digits that occur
- **MSD-2** `NiceSearch.width_recurrence`: width recurrence `diff_j = b·diff_{j+1} + (yd_j − xd_j)` is exact; once `diff ≥ b−1` every lower position is too (`width_ge_of_succ`)
- **MSD-3** `NiceSearch.powerDomains_sound`: dropping a power's domains (unequal digit counts) is sound
- **MSD-4** `NiceSearch.no_nice_of_not_hasSDR`: Hall soundness: no injective digit choice ⇒ no nice n in the range; model form `no_nice_of_analyzeRange` for the executable `analyzeRange`
- **MSD-10** `NiceSearch.sdrClosure_iff`: singleton closure preserves SDR existence: a one-digit domain `{x}` takes `x`, which then leaves every other domain, in both directions (`hasSDR_cons_singleton_iff`); the order of the constraints does not matter (`hasSDR_perm`) and an emptied domain has no SDR (`not_hasSDR_of_empty`); so the closure decides exactly whether an SDR exists (`sdrClosure_iff`)
- **MSD-6** `NiceSearch.Sound.sublist`: domain-slot overflow only drops constraints
- **MSD-7** `NiceSearch.validRanges_cover`: recursive subdivision (factor 2, depth fuel, floor): every nice n of the input lies in some emitted leaf; leaves are sub-intervals (`validRanges_subset`)
- **MSD-8** `NiceSearch.no_nice_of_equal_singletons`: the over-64 prefix path is the singleton-domain case of MSD-4
- **MSD-9** `NiceSearch.analyzeRange_mono`: monotone rejection: `Rejected(I) ∧ J ⊆ I → Rejected(J)` (sub-range domains are position-wise subsets, `rangeDomains_sub`; an SDR transfers, `hasSDR_of_sub`); certificates grow on sub-ranges (`fixedDigits_sub`)
- **CRS-1** `NiceSearch.no_nice_of_cross`: singleton high digit at position `≥ k` colliding with a residue's exact low digit kills the residue in the range
- **CRS-2** `NiceSearch.validRangesMasked_cover`: the masked recursion: every nice n lies in a leaf whose inherited mask consists of high digits of n (a certificate for a range holds on every sub-range)
- **CRS-3** `NiceSearch.validRangesMasked_cover`: an empty or partial certificate is sound (mask soundness holds for any accumulated mask, so ignoring certificates only checks more candidates)
- **REF-1** `NiceSearch.msd_lsd_skip_unsound`: the removed MSD×LSD skip is unsound: witness b=10, k=2, `[68,70)` (quotient test passes, low digits differ, 69 is nice); generally `n mod b^k` is never constant on a range of size > 1 (`mod_pow_not_constant`)
- **AFF-1** `NiceSearch.affine_mid_digit`: for `n = s + b^k·t` and `i < m ≤ k`, digit `k + i` of `n²` (of `n³`) is digit `i` of `(⌊s²/b^k⌋ + 2st) mod b^m` (of `(⌊s³/b^k⌋ + 3s²t) mod b^m`); the filter keeps a nice number whose middle positions are real digits and whose known digits sit below `k` or at `2k` and above (`affineSurvives_of_isNice`); at or above `b^(2k−1)` the certificate of a range of two or more numbers holds no position below `2k` (`highDigit_two_of_mem_fixedDigits`)
- **END-1** `NiceSearch.niceonly_complete`: the modelled niceonly pipeline (masked subdivision × stride walk × one-AND × affine stage × nice check) reports every nice n of the range (`niceonly_complete`, for b ≥ 6, k ≤ 3, floor ≥ 1) and only nice n of the range (`niceonly_sound`)
- **GPU-1** `NiceSearch.blockTiling_cover`: block tiling (64-chunk blocks, descending powers of two, partial chunk) sums to the field size and covers it without overlap (`blockLens_sum`, `tile_cover`, `tile_disjoint`)
- **GPU-2** `NiceSearch.validRangesMasked_block`: block starts yield the same leaves and masks as chunk starts: with chunks wider than the floor and the block given j extra depth levels, the masked recursion on a 2^j-chunk block equals the concatenation of the per-chunk recursions (`validRangesMasked_block`; uses MSD-9 and `fixedDigits_sub`)
- **GPU-3** `NiceSearch.validRangesMasked_cover`: mixing floors within a field loses nothing: the cover theorem holds for every floor and depth, so any per-block choice is sound
- **GPU-4** `NiceSearch.lane_partition`: lane tiling partitions the ordinals for any lane count
- **GPU-5** `NiceSearch.split16_exact`: split16 chunk step is exact when `d < 2^16`
- **GPU-6** `NiceSearch.truncated_mul_mod`: dropping partial products at or above limb L is reduction mod B^L (`truncated_mul_mod`); a limb step `a·c + acc + carry` stays below 2^32 for B ≤ 2^16 (`limb_step_lt`)
- **GPU-7** `NiceSearch.hornerMod_chunksBE`: chunked Horner over the base-`2^c` chunks computes `off mod M` (`hornerMod_chunksBE`) and each step stays below `2^32` while `M ≤ 2^(32−c)` (`horner_step_lt`)
- **GPU-8** `NiceSearch.prefilter_rejects_all_of_short`: prefilter = LSD-2 at depth p (`mem_lsdBitmap_of_isNice`); where neither power has p digits the zero padding rejects every candidate (`prefilter_rejects_all_of_short`, the v3.2.14 failure)
- **GPU-9** `NiceSearch.mod_m_split`: `mod_m` via `2^64 mod M` is correct under NUM-8
- **FLD-1** `NiceSearch.inField_iff`: fields partition the base (`inField_iff`); a field lies inside the chunk containing its start point (`field_subset_chunk`, `chunk_of_field_start`), which is what start-point chunk matching relies on
- **DET-1** `NiceSearch.histogram_fold`: histogram bins fold batch by batch: the count of a value over a concatenation is the sum of the counts
- **DET-1b** `NiceSearch.topN_of_superset`: top-N compaction drops nothing: an element with fewer than N strictly larger keys in the whole list has fewer than N in any batch (ties not modelled)
- **THY-2** `NiceSearch.collapse_of_invariant`: carry-blind collapse: a linear digit statistic invariant under every carry move has `w_{i+1} ≡ b·w_i` (`weight_rel_of_invariant`) and equals `w₀·N (mod m)` (`collapse`)
- **THY-3** `NiceSearch.complement_sum`: once some output digits are fixed, the rest sum to the complement and form the complement set (`complement_set`): a digit-sum window on the unassigned positions is vacuous
- **THY-4** `NiceSearch.sieve_b2_complete`: the `b²−1` sieve on `n² + n³` adds nothing to casting out `b−1`s when both powers have at least 3 digits (every nice band with `b ≥ 6`, RNG-3): every residue mod `b²−1` that the digit-sum congruence allows is `(x + y) mod (b²−1)` for digit words `x`, `y` of those lengths with nonzero leading digits that use every digit once (`sieve_b2_complete`; the `c`-subset sums of `0..b−1` fill an interval, `exists_subset_sum`). False without the length condition: base 4, lengths (2, 2) (Janzert's `base_four_sieve_is_incomplete`)
- **THY-5** `NiceSearch.window_sound`: middle-window filter is sound (digits at `p..p+w` depend on `n mod b^(p+w)`)
- **THY-6** `NiceSearch.hall_relaxation_incomplete`: the interval-domain Hall check is strictly incomplete: base 10, `[47, 60]` has an SDR but no nice number (by `decide`)
- **THY-9** `NiceSearch.witnessRate`: witness model `λ_b = range_b · b!/b^b` (definition only)
- **JOIN-1** `NiceSearch.highDigit_of_cert`: top certificate: the digits common to `a^j` and `e^j` at positions `≥ cap`, from the top down to the first disagreement (`x/b^i = y/b^i`), are digits of `n^j` at positions at or above `cert_floor` for every `n` in `[a, e]` (`highDigit_of_cert`); a repeat among them rules the interval out (`not_isNice_of_cert_none`); the top layer keeps every nice number's prefix (`mem_topLayer_of_isNice`)
- **JOIN-2** `NiceSearch.mem_bottomList_of_isNice`: bottom side: `bot_dfs`'s digit step is AFF-1 with one digit (`botDigits_eq`), and the search keeps every nice number's residue mod `b^k` with its exact low output digits, the partition's digits forced (`mem_bottomList_of_isNice`)
- **JOIN-3** `NiceSearch.disjoint_high_lowSet`: the AND: a nice number's high digits (positions `≥ k`) and its low digits (positions `< k`) are disjoint, so its pair passes
- **JOIN-4** `NiceSearch.joinSlices_flatMap`: the slices concatenate to the field, in order, so every number is in exactly one
- **JOIN-5** `NiceSearch.mem_joinPartition_of_isNice`: every `n` is exactly one (prefix, low part) pair (`join_pair_unique`); the join finds each nice `n` of the field in partition `⌊n/b^f0⌋ mod b^p`: its prefix's ancestor is in the top layer, its certificate passes, its residue is in the bottom list, the probe of root `n mod (b−1)` reaches its bucket (key digits, digit-sum class `class_eq`) and the AND passes (`mem_joinPartition_of_isNice`)
- **JOIN-6** `NiceSearch.prefilterAt_of_isNice`: the prefilter keeps every nice `n` whose powers have `k2` digits: its low output digits are distinct, and the certificate is tested only when the floor the Rust uses is at least `k2` (a whole block's `full_floor` is at most its own floor, `certFloor_block_mono`)
- **END-2** `NiceSearch.joinField_complete`: the modelled overlap join (slices × partitions × top layer × bottom search × probe × AND × prefilter × nice check) reports every nice n of a field of `L`-digit numbers (`joinField_complete`, for `t ≤ L`, `k < L < t + k`, `p ≤ t + k − L` and a prefilter depth `s²` reaches; `joinField_slices_complete` with the client's slicing) and only nice n of the field (`joinField_sound`)
<!-- status:end -->
