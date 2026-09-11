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
independent of the code. The end-to-end target is `END-1`: the modelled
niceonly pipeline reports every nice number in a field.

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
and diffs (about 9,000 checks, ~10 s). Theorems pin "model = spec"; this
pins "model = code". The fixtures are checked in; regenerate them when
the Rust tables change.

## The registry and the Rust tags

`CLAIMS.md` has one row per claim. A Rust doc comment of the form

```rust
/// Lean: `Nice.mem_residueFilter_of_isNice` (RES-1)
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
| 2 | 0 | 10 | 0 | 0 | 0 |
| 3 | 0 | 8 | 0 | 1 | 0 |
| 4 | 0 | 4 | 0 | 0 | 1 |
| 5 | 0 | 12 | 0 | 0 | 0 |
| 6 | 1 | 4 | 0 | 4 | 0 |
| — | 0 | 0 | 0 | 0 | 2 |

Proved or defined so far:

- **DEF-1** `Nice.IsNice`: `IsNice b n` ⇔ the base-b digits of n² followed by those of n³ permute `0..b-1`
- **DEF-1a** `Nice.isNice_iff_pandigital`: the three-part `Pandigital` definition of `origin/proofs` is equivalent
- **DEF-2** `Nice.numUniques_eq_iff_isNice`: inside the base range, `numUniques b n = b ↔ IsNice b n`
- **DEF-3** `Nice.one_le_numUniques`: `1 ≤ numUniques b n` for `n ≥ 1` (histogram bin 0 is empty)
- **DEF-4** `Nice.nearMissCutoff`: near-miss cutoff is `⌊0.9 b⌋` with strict `>` (`IsNearMiss`); every nice number is a near miss (`isNearMiss_of_isNice`)
- **RNG-1** `Nice.nice_digit_count`: `IsNice b n → numDigits(n²) + numDigits(n³) = b`
- **RNG-2** `Nice.memBaseRange_of_inBaseRange`: the per-`b mod 5` closed-form interval contains every n with `numDigits(n²) + numDigits(n³) = b`
- **RNG-2b** `Nice.inBaseRange_of_memBaseRange`: conversely every n in the closed-form interval has digit-count sum b; with RNG-2 the interval is exactly the search range (`memBaseRange_iff`)
- **RNG-3** `Nice.three_le_numDigits_of_inBaseRange`: inside the range every power has `≥ k` digits for `k ≤ 3`, `b ≥ 6`
- **RNG-4** `Nice.not_inBaseRange_of_one_mod_five`: `b ≡ 1 (mod 5)` ⇒ no n has digit-count sum b
- **RNG-5** `Nice.numDigits_pow_mono`: `numDigits b (n^e)` is monotone in n
- **NUM-1** `Nice.Const.u128_cutoff_40`: `(rangeEnd 40 − 1)^3 < 2^128`
- **NUM-2** `Nice.Const.u256_cutoff`: `(rangeEnd b − 1)^3 < 2^256` for `b ≤ 68`; 69 fits, 70 does not
- **NUM-3** `Nice.Const.max_fw_digits`: `numDigits b (n^3) ≤ 38` for `b ≤ 64` in range
- **NUM-4** `Nice.Const.stride_modulus_u32`: `(b−1)·b^3 < 2^32` for `b ≤ 256` (u32 stride table)
- **NUM-4a** `Nice.Const.stride_modulus_gpu`: `(b−1)·b^3 < 2^28` for `b ≤ 128` (`MAX_STRIDE_MODULUS`)
- **NUM-5** `Nice.Const.mask_width`: digit masks need `b ≤ 64` (u64) / `b ≤ 128` (two words)
- **NUM-6** `Nice.chunkExp_spec`: GPU chunk constants: `b^e < bound ≤ b^(e+1)` for `e = chunkExp b bound` (maximal), and the split16 shift bound `(div−1)·2^16 < 2^32` (`split16_shift_bound`)
- **NUM-7** `Nice.prefilter_sound`: the prefilter depth `min(numDigits(start²), numDigits(start³)) − 1` computed exactly from the range start is sound for every candidate at or after it
- **NUM-8** `Nice.Const.mod_m_bound`: `M² + M < 2^64` for `M < 2^32`
- **NUM-9** `Nice.Const.histogram_bins`: histogram bins cannot overflow u32
- **RES-1** `Nice.mem_residueFilter_of_isNice`: `IsNice b n → n² + n³ ≡ b(b−1)/2 (mod b−1)`; `n mod (b−1) ∈ residueFilter b`
- **RES-1a** `Nice.nice_digit_sum`: a nice number's output digits sum to `b(b−1)/2`
- **RES-2** `Nice.no_nice_of_three_mod_four`: `b ≡ 3 (mod 4) → ∀ n, ¬IsNice b n`
- **RES-3** `Nice.no_nice_of_residueFilter_empty`: `residueFilter b = ∅ → ∀ n, ¬IsNice b n`; `residueFilter 11 = ∅`
- **LSD-1** `Nice.digit_pow_mod_pow`: `digit b (n^e) j` for `j < k` depends only on `n mod b^k`
- **LSD-2** `Nice.mem_lsdBitmap_of_isNice`: nice + RNG-3 ⇒ the 2k fixed-width low digits are pairwise distinct ⇒ `n mod b^k ∈ lsdBitmap b k`
- **STR-1** `Nice.mem_validResidues_iff`: `Coprime (b−1) (b^k)`; passes both ⇔ `n mod M ∈ validResidues`
- **STR-2** `Nice.walk_eq_filter`: the gap-table walk visits exactly the valid n in `[start,end)` in order
- **STR-3** `Nice.seeded_iff_isNice`: seeded check equals the plain check under RNG-3
- **STR-4** `Nice.lowMask_eq`: `low_digit_masks[i]` is exactly the low-digit set of residue i's powers
- **GPU-0** `Nice.exists_ordinal_eq`: ordinal formula `B0 + ⌊g/#V⌋·M + V[g mod #V]` is strictly increasing (`ordinal_strictMono`), always valid (`ordinal_valid`) and hits every valid n ≥ B0, so it enumerates the same set as STR-2
- **MSD-1** `Nice.digit_mem_cyclicInterval`: interval digit domains are a superset of the digits that occur
- **MSD-2** `Nice.width_recurrence`: width recurrence `diff_j = b·diff_{j+1} + (yd_j − xd_j)` is exact; once `diff ≥ b−1` every lower position is too (`width_ge_of_succ`)
- **MSD-3** `Nice.powerDomains_sound`: dropping a power's domains (unequal digit counts) is sound
- **MSD-4** `Nice.no_nice_of_not_hasSDR`: Hall soundness: no injective digit choice ⇒ no nice n in the range; model form `no_nice_of_analyzeRange` for the executable `analyzeRange`
- **MSD-6** `Nice.Sound.sublist`: domain-slot overflow only drops constraints
- **MSD-7** `Nice.validRanges_cover`: recursive subdivision (factor 2, depth fuel, floor): every nice n of the input lies in some emitted leaf; leaves are sub-intervals (`validRanges_subset`)
- **MSD-8** `Nice.no_nice_of_equal_singletons`: the over-64 prefix path is the singleton-domain case of MSD-4
- **MSD-9** `Nice.analyzeRange_mono`: monotone rejection: `Rejected(I) ∧ J ⊆ I → Rejected(J)` (sub-range domains are position-wise subsets, `rangeDomains_sub`; an SDR transfers, `hasSDR_of_sub`); certificates grow on sub-ranges (`fixedDigits_sub`)
- **CRS-1** `Nice.no_nice_of_cross`: singleton high digit at position `≥ k` colliding with a residue's exact low digit kills the residue in the range
- **CRS-2** `Nice.validRangesMasked_cover`: the masked recursion: every nice n lies in a leaf whose inherited mask consists of high digits of n (a certificate for a range holds on every sub-range)
- **CRS-3** `Nice.validRangesMasked_cover`: an empty or partial certificate is sound (mask soundness holds for any accumulated mask, so ignoring certificates only checks more candidates)
- **REF-1** `Nice.msd_lsd_skip_unsound`: the removed MSD×LSD skip is unsound: witness b=10, k=2, `[68,70)` (quotient test passes, low digits differ, 69 is nice); generally `n mod b^k` is never constant on a range of size > 1 (`mod_pow_not_constant`)
- **END-1** `Nice.niceonly_complete`: the modelled niceonly pipeline (masked subdivision × stride walk × one-AND × nice check) reports every nice n of the range (`niceonly_complete`, for b ≥ 6, k ≤ 3) and only nice n of the range (`niceonly_sound`)
- **GPU-1** `Nice.blockTiling_cover`: block tiling (64-chunk blocks, descending powers of two, partial chunk) sums to the field size and covers it without overlap (`blockLens_sum`, `tile_cover`, `tile_disjoint`)
- **GPU-2** `Nice.validRangesMasked_block`: block starts yield the same leaves and masks as chunk starts: with chunks wider than the floor and the block given j extra depth levels, the masked recursion on a 2^j-chunk block equals the concatenation of the per-chunk recursions (`validRangesMasked_block`; uses MSD-9 and `fixedDigits_sub`)
- **GPU-3** `Nice.validRangesMasked_cover`: mixing floors within a field loses nothing: the cover theorem holds for every floor and depth, so any per-block choice is sound
- **GPU-4** `Nice.lane_partition`: lane tiling partitions the ordinals for any lane count
- **GPU-5** `Nice.split16_exact`: split16 chunk step is exact when `d < 2^16`
- **GPU-6** `Nice.truncated_mul_mod`: dropping partial products at or above limb L is reduction mod B^L (`truncated_mul_mod`); a limb step `a·c + acc + carry` stays below 2^32 for B ≤ 2^16 (`limb_step_lt`)
- **GPU-7** `Nice.hornerMod_chunksBE`: chunked Horner over the base-`2^c` chunks computes `off mod M` (`hornerMod_chunksBE`) and each step stays below `2^32` while `M ≤ 2^(32−c)` (`horner_step_lt`)
- **GPU-8** `Nice.prefilter_rejects_all_of_short`: prefilter = LSD-2 at depth p (`mem_lsdBitmap_of_isNice`); where neither power has p digits the zero padding rejects every candidate (`prefilter_rejects_all_of_short`, the v3.2.14 failure)
- **GPU-9** `Nice.mod_m_split`: `mod_m` via `2^64 mod M` is correct under NUM-8
- **FLD-1** `Nice.inField_iff`: fields partition the base (`inField_iff`); a field lies inside the chunk containing its start point (`field_subset_chunk`, `chunk_of_field_start`), which is what start-point chunk matching relies on
- **DET-1** `Nice.histogram_fold`: histogram bins fold batch by batch: the count of a value over a concatenation is the sum of the counts
- **DET-1b** `Nice.topN_of_superset`: top-N compaction drops nothing: an element with fewer than N strictly larger keys in the whole list has fewer than N in any batch (ties not modelled)
- **THY-2** `Nice.collapse_of_invariant`: carry-blind collapse: a linear digit statistic invariant under every carry move has `w_{i+1} ≡ b·w_i` (`weight_rel_of_invariant`) and equals `w₀·N (mod m)` (`collapse`)
- **THY-3** `Nice.complement_sum`: once some output digits are fixed, the rest sum to the complement and form the complement set (`complement_set`): a digit-sum window on the unassigned positions is vacuous
- **THY-5** `Nice.window_sound`: middle-window filter is sound (digits at `p..p+w` depend on `n mod b^(p+w)`)
- **THY-6** `Nice.hall_relaxation_incomplete`: the interval-domain Hall check is strictly incomplete: base 10, `[47, 60]` has an SDR but no nice number (by `decide`)
- **THY-9** `Nice.witnessRate`: witness model `λ_b = range_b · b!/b^b` (definition only)
<!-- status:end -->
