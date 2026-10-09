# Claims registry

One row per mathematical claim the search relies on, links to the Lean
declaration that states or proves it and to the Rust sites that depend on
it. `scripts/check_claims.py` validates this file against the built Lean
library and the `Lean:` tags in the Rust sources; the README's status
table is generated from it. Phases are those of `DESIGN.md`.

Columns: `status` ∈ `def` (a definition), `proved` (sorry-free, standard
axioms only), `stated` (declared, proof incomplete), `planned` (no Lean yet;
the `lean` column is the intended name), `conjecture` (in
`Nice/Conjectures.lean`), `refuted` (in `Nice/Refuted.lean`), or one of
`rust-test` / `device-test` / `sql-audit` for claims that are deliberately
checked outside Lean. `evidence` is what the Rust side has today.

<!-- claims:begin -->
| id | statement | lean | rust | evidence | status | phase |
|---|---|---|---|---|---|---|
| DEF-1 | `IsNice b n` ⇔ the base-b digits of n² followed by those of n³ permute `0..b-1` | `NiceSearch.IsNice` | `common/src/client_process.rs::get_is_nice` | tests | def | 0 |
| DEF-1a | the three-part `Pandigital` definition of `origin/proofs` is equivalent | `NiceSearch.isNice_iff_pandigital` | — | — | proved | 0 |
| DEF-2 | inside the base range, `numUniques b n = b ↔ IsNice b n` | `NiceSearch.numUniques_eq_iff_isNice` | `common/src/client_process.rs::get_num_unique_digits` | comment | proved | 1 |
| DEF-3 | `1 ≤ numUniques b n` for `n ≥ 1` (histogram bin 0 is empty) | `NiceSearch.one_le_numUniques` | `common/src/distribution_stats.rs` | comment | proved | 1 |
| DEF-4 | near-miss cutoff is `⌊0.9 b⌋` with strict `>` (`IsNearMiss`); every nice number is a near miss (`isNearMiss_of_isNice`) | `NiceSearch.nearMissCutoff` | `common/src/number_stats.rs::get_near_miss_cutoff` | tests | def | 1 |
| RNG-1 | `IsNice b n → numDigits(n²) + numDigits(n³) = b` | `NiceSearch.nice_digit_count` | `common/src/base_range.rs` | tests | proved | 0 |
| RNG-2 | the per-`b mod 5` closed-form interval contains every n with `numDigits(n²) + numDigits(n³) = b` | `NiceSearch.memBaseRange_of_inBaseRange` | `common/src/base_range.rs::get_base_range_natural` | tests pin 8 bases | proved | 1 |
| RNG-2b | conversely every n in the closed-form interval has digit-count sum b; with RNG-2 the interval is exactly the search range (`memBaseRange_iff`) | `NiceSearch.inBaseRange_of_memBaseRange` | `common/src/base_range.rs::get_base_range_natural` | tests pin 8 bases | proved | 1 |
| RNG-3 | inside the range every power has `≥ k` digits for `k ≤ 3`, `b ≥ 6` | `NiceSearch.three_le_numDigits_of_inBaseRange` | `common/src/lsd_filter.rs`, `common/src/client_process.rs::get_is_nice_with_known_lsd` | comment | proved | 1 |
| RNG-4 | `b ≡ 1 (mod 5)` ⇒ no n has digit-count sum b | `NiceSearch.not_inBaseRange_of_one_mod_five` | `common/src/base_range.rs` | asserted | proved | 1 |
| RNG-5 | `numDigits b (n^e)` is monotone in n | `NiceSearch.numDigits_pow_mono` | `common/src/gpu_config.rs::prefilter_params` | comment | proved | 1 |
| NUM-1 | `(rangeEnd 40 − 1)^3 < 2^128` | `NiceSearch.Const.u128_cutoff_40` | `common/src/client_process.rs::MAX_BASE_FOR_FIXED_WIDTH_U128` | rust test (PR #158) | proved | 1 |
| NUM-2 | `(rangeEnd b − 1)^3 < 2^256` for `b ≤ 68`; 69 fits, 70 does not | `NiceSearch.Const.u256_cutoff` | `common/src/client_process.rs::MAX_BASE_FOR_FIXED_WIDTH_U256` | rust test (PR #158) | proved | 1 |
| NUM-3 | `numDigits b (n^3) ≤ 38` for `b ≤ 64` in range | `NiceSearch.Const.max_fw_digits` | `common/src/msd_prefix_filter.rs::MAX_FW_DIGITS` | comment | proved | 1 |
| NUM-4 | `(b−1)·b^3 < 2^32` for `b ≤ 256` (u32 stride table) | `NiceSearch.Const.stride_modulus_u32` | `common/src/stride_filter.rs::StrideTable::new` | runtime assert | proved | 1 |
| NUM-4a | `(b−1)·b^3 < 2^28` for `b ≤ 128` (`MAX_STRIDE_MODULUS`) | `NiceSearch.Const.stride_modulus_gpu` | `common/src/gpu_niceonly.rs::MAX_STRIDE_MODULUS` | runtime assert | proved | 1 |
| NUM-5 | digit masks need `b ≤ 64` (u64) / `b ≤ 128` (two words) | `NiceSearch.Const.mask_width` | `common/src/stride_filter.rs::StrideTable::low_digit_masks` | const-assert | proved | 1 |
| NUM-6 | chunk constants: `b^e < bound ≤ b^(e+1)` for `e = chunkExp b bound` (maximal), for the GPU chunks (bounds `2^31`, `2^16`) and the MSD endpoint extraction's chunks (`chunk_digits`, bound `2^32`); and the split16 shift bound `(div−1)·2^16 < 2^32` (`split16_shift_bound`) | `NiceSearch.chunkExp_spec` | `common/src/gpu_config.rs::chunk_constants`, `common/src/msd_prefix_filter.rs::chunk_digits` | tests (extraction at the chunk boundary for `chunk_digits`) | proved | 1 |
| NUM-7 | the prefilter depth `min(numDigits(start²), numDigits(start³)) − 1` computed exactly from the range start is sound for every candidate at or after it | `NiceSearch.prefilter_sound` | `common/src/gpu_config.rs::prefilter_params` | integer test re-checks the float | proved | 1 |
| NUM-8 | `M² + M < 2^64` for `M < 2^32` | `NiceSearch.Const.mod_m_bound` | `common/src/cuda/nice_kernels.cu::mod_m` | comment (fixed PR #158) | proved | 1 |
| NUM-9 | histogram bins cannot overflow u32 | `NiceSearch.Const.histogram_bins` | `common/src/cubecl_backend.rs::DRAIN_INTERVAL` | const-assert + test | proved | 1 |
| RES-1 | `IsNice b n → n² + n³ ≡ b(b−1)/2 (mod b−1)`; `n mod (b−1) ∈ residueFilter b` | `NiceSearch.mem_residueFilter_of_isNice` | `common/src/residue_filter.rs::get_residue_filter` | tests | proved | 2 |
| RES-1a | a nice number's output digits sum to `b(b−1)/2` | `NiceSearch.nice_digit_sum` | `common/src/residue_filter.rs` | — | proved | 0 |
| RES-2 | `b ≡ 3 (mod 4) → ∀ n, ¬IsNice b n` | `NiceSearch.no_nice_of_three_mod_four` | `common/src/residue_filter.rs` (oracle test) | oracle test 5–512 | proved | 2 |
| RES-3 | `residueFilter b = ∅ → ∀ n, ¬IsNice b n`; `residueFilter 11 = ∅` | `NiceSearch.no_nice_of_residueFilter_empty` | `common/src/gpu_niceonly.rs::residue_empty_result` | tests | proved | 2 |
| LSD-1 | `digit b (n^e) j` for `j < k` depends only on `n mod b^k` | `NiceSearch.digit_pow_mod_pow` | `common/src/lsd_filter.rs` | comment | proved | 2 |
| LSD-2 | nice + RNG-3 ⇒ the 2k fixed-width low digits are pairwise distinct ⇒ `n mod b^k ∈ lsdBitmap b k` | `NiceSearch.mem_lsdBitmap_of_isNice` | `common/src/lsd_filter.rs::get_valid_multi_lsd_bitmap` | brute-force b 4–16 | proved | 2 |
| STR-1 | `Coprime (b−1) (b^k)`; passes both ⇔ `n mod M ∈ validResidues` | `NiceSearch.mem_validResidues_iff` | `common/src/stride_filter.rs::StrideTable::new` | comment | proved | 2 |
| STR-2 | the gap-table walk visits exactly the valid n in `[start,end)` in order | `NiceSearch.walk_eq_filter` | `common/src/stride_filter.rs::iterate_range_masked` | tests (gap sum = M) | proved | 2 |
| STR-3 | seeded check equals the plain check under RNG-3 | `NiceSearch.seeded_iff_isNice` | `common/src/client_process.rs::get_is_nice_with_known_lsd` | 5000-sample test | proved | 2 |
| STR-4 | `low_digit_masks[i]` is exactly the low-digit set of residue i's powers | `NiceSearch.lowMask_eq` | `common/src/stride_filter.rs::StrideTable::new` | by construction | proved | 2 |
| GPU-0 | ordinal formula `B0 + ⌊g/#V⌋·M + V[g mod #V]` is strictly increasing (`ordinal_strictMono`), always valid (`ordinal_valid`) and hits every valid n ≥ B0, so it enumerates the same set as STR-2 | `NiceSearch.exists_ordinal_eq` | `common/src/cuda/nice_kernels.cu`, `common/src/vulkan/codegen.rs`, `common/src/cubecl_backend.rs` | host-mirror tests | proved | 2 |
| MSD-1 | interval digit domains are a superset of the digits that occur | `NiceSearch.digit_mem_cyclicInterval` | `common/src/msd_prefix_filter.rs::collect_power_domains` | tests | proved | 3 |
| MSD-2 | width recurrence `diff_j = b·diff_{j+1} + (yd_j − xd_j)` is exact; once `diff ≥ b−1` every lower position is too (`width_ge_of_succ`) | `NiceSearch.width_recurrence` | `common/src/msd_prefix_filter.rs::collect_power_domains` | comment | proved | 3 |
| MSD-3 | dropping a power's domains (unequal digit counts) is sound | `NiceSearch.powerDomains_sound` | `common/src/msd_prefix_filter.rs::analyze_msd_prefix` | comment | proved | 3 |
| MSD-4 | Hall soundness: no injective digit choice ⇒ no nice n in the range; model form `no_nice_of_analyzeRange` for the executable `analyzeRange` | `NiceSearch.no_nice_of_not_hasSDR` | `common/src/msd_prefix_filter.rs::has_distinct_assignment`, `has_distinct_assignment_closure` | brute-force b 4–16 | proved | 3 |
| MSD-5 | Kuhn completeness: the augmenting-path search returns false only when no SDR exists | `NiceSearch.Model.Hall.kuhn_complete` | `common/src/msd_prefix_filter.rs::hall_augment` | oracle test (PR #157) | planned | 3 |
| MSD-10 | singleton closure preserves SDR existence: a one-digit domain `{x}` takes `x`, which then leaves every other domain, in both directions (`hasSDR_cons_singleton_iff`); the order of the constraints does not matter (`hasSDR_perm`) and an emptied domain has no SDR (`not_hasSDR_of_empty`); so the closure decides exactly whether an SDR exists (`sdrClosure_iff`) | `NiceSearch.sdrClosure_iff` | `common/src/msd_prefix_filter.rs::has_distinct_assignment_closure` | closure vs Kuhn vs brute-force Hall on 20k sets (`closure_matching_agrees_with_kuhn`) | proved | 3 |
| MSD-6 | domain-slot overflow only drops constraints | `NiceSearch.Sound.sublist` | `common/src/msd_prefix_filter.rs::collect_power_domains` | comment | proved | 3 |
| MSD-7 | recursive subdivision (factor 2, depth fuel, floor): every nice n of the input lies in some emitted leaf; leaves are sub-intervals (`validRanges_subset`) | `NiceSearch.validRanges_cover` | `common/src/msd_prefix_filter.rs::get_valid_ranges_recursive`, `recurse_generic`, `recurse_fw` | brute-force b 4–16 | proved | 3 |
| MSD-8 | the over-64 prefix path is the singleton-domain case of MSD-4 | `NiceSearch.no_nice_of_equal_singletons` | `common/src/msd_prefix_filter.rs::analyze_range_over_64` | comment | proved | 3 |
| MSD-9 | monotone rejection: `Rejected(I) ∧ J ⊆ I → Rejected(J)` (sub-range domains are position-wise subsets, `rangeDomains_sub`; an SDR transfers, `hasSDR_of_sub`); certificates grow on sub-ranges (`fixedDigits_sub`) | `NiceSearch.analyzeRange_mono` | `common/src/gpu_niceonly.rs::BlockTiling` | one-window test | proved | 3 |
| CRS-1 | singleton high digit at position `≥ k` colliding with a residue's exact low digit kills the residue in the range | `NiceSearch.no_nice_of_cross` | `common/src/msd_prefix_filter.rs::analyze_range`, `common/src/stride_filter.rs::iterate_range_masked` | tests (b10/17/22/25) | proved | 4 |
| CRS-2 | the masked recursion: every nice n lies in a leaf whose inherited mask consists of high digits of n (a certificate for a range holds on every sub-range) | `NiceSearch.validRangesMasked_cover` | `common/src/msd_prefix_filter.rs::recurse_generic`, `recurse_fw` | identical-leaves test | proved | 4 |
| CRS-3 | an empty or partial certificate is sound (mask soundness holds for any accumulated mask, so ignoring certificates only checks more candidates) | `NiceSearch.validRangesMasked_cover` | `common/src/vulkan/mod.rs` | comment | proved | 4 |
| REF-1 | the removed MSD×LSD skip is unsound: witness b=10, k=2, `[68,70)` (quotient test passes, low digits differ, 69 is nice); generally `n mod b^k` is never constant on a range of size > 1 (`mod_pow_not_constant`) | `NiceSearch.msd_lsd_skip_unsound` | `common/src/msd_prefix_filter.rs` (NOTE) | regression test | refuted | 4 |
| END-1 | the modelled niceonly pipeline (masked subdivision × stride walk × one-AND × nice check) reports every nice n of the range (`niceonly_complete`, for b ≥ 6, k ≤ 3) and only nice n of the range (`niceonly_sound`) | `NiceSearch.niceonly_complete` | `common/src/client_process.rs::process_range_niceonly` | small-base brute force | proved | 4 |
| GPU-1 | block tiling (64-chunk blocks, descending powers of two, partial chunk) sums to the field size and covers it without overlap (`blockLens_sum`, `tile_cover`, `tile_disjoint`) | `NiceSearch.blockTiling_cover` | `common/src/gpu_niceonly.rs::BlockTiling::new` | test | proved | 5 |
| GPU-2 | block starts yield the same leaves and masks as chunk starts: with chunks wider than the floor and the block given j extra depth levels, the masked recursion on a 2^j-chunk block equals the concatenation of the per-chunk recursions (`validRangesMasked_block`; uses MSD-9 and `fixedDigits_sub`) | `NiceSearch.validRangesMasked_block` | `common/src/gpu_niceonly.rs` | one test | proved | 5 |
| GPU-3 | mixing floors within a field loses nothing: the cover theorem holds for every floor and depth, so any per-block choice is sound | `NiceSearch.validRangesMasked_cover` | `common/src/gpu_niceonly.rs` | comment | proved | 5 |
| GPU-4 | lane tiling partitions the ordinals for any lane count | `NiceSearch.lane_partition` | `common/src/gpu_niceonly.rs::lane_shift_for` | device tests | proved | 5 |
| GPU-5 | split16 chunk step is exact when `d < 2^16` | `NiceSearch.split16_exact` | `common/src/vulkan/codegen.rs` | test | proved | 5 |
| GPU-6 | dropping partial products at or above limb L is reduction mod B^L (`truncated_mul_mod`); a limb step `a·c + acc + carry` stays below 2^32 for B ≤ 2^16 (`limb_step_lt`) | `NiceSearch.truncated_mul_mod` | `common/src/vulkan/codegen.rs`, `common/src/gpu_config.rs` | tests | proved | 5 |
| GPU-7 | chunked Horner over the base-`2^c` chunks computes `off mod M` (`hornerMod_chunksBE`) and each step stays below `2^32` while `M ≤ 2^(32−c)` (`horner_step_lt`) | `NiceSearch.hornerMod_chunksBE` | `common/src/gpu_niceonly.rs::stride_chunk_bits` | tests | proved | 5 |
| GPU-8 | prefilter = LSD-2 at depth p (`mem_lsdBitmap_of_isNice`); where neither power has p digits the zero padding rejects every candidate (`prefilter_rejects_all_of_short`, the v3.2.14 failure) | `NiceSearch.prefilter_rejects_all_of_short` | `common/src/gpu_config.rs::prefilter_params` | tests | proved | 5 |
| GPU-9 | `mod_m` via `2^64 mod M` is correct under NUM-8 | `NiceSearch.mod_m_split` | `common/src/cuda/nice_kernels.cu::mod_m` | test | proved | 5 |
| GPU-C | warp/cube compaction queue bounds and uniformity | — | `common/src/cuda/nice_kernels.cu`, `common/src/cubecl_backend.rs` | device tests | device-test | — |
| FLD-1 | fields partition the base (`inField_iff`); a field lies inside the chunk containing its start point (`field_subset_chunk`, `chunk_of_field_start`), which is what start-point chunk matching relies on | `NiceSearch.inField_iff` | `common/src/generate_fields.rs`, `common/src/generate_chunks.rs`, `common/src/db_util/chunks.rs::reassign_fields_to_chunks` | tests | proved | 5 |
| FLD-DB | the stored field/chunk rows partition each base | — | `common/src/db_util/audit.rs` (PR #159) | sql audit | sql-audit | — |
| DET-1 | histogram bins fold batch by batch: the count of a value over a concatenation is the sum of the counts | `NiceSearch.histogram_fold` | `common/src/distribution_stats.rs::DistributionAccumulator` | tests | proved | 5 |
| DET-1b | top-N compaction drops nothing: an element with fewer than N strictly larger keys in the whole list has fewer than N in any batch (ties not modelled) | `NiceSearch.topN_of_superset` | `common/src/number_stats.rs::NumbersAccumulator` | tests | proved | 5 |
| THY-1 | residue-count closed form over the prime powers of `b−1` | `NiceSearch.Theory.residueFilter_card` | `common/src/residue_filter.rs` (oracle test) | oracle test 5–512 | planned | 6 |
| THY-2 | carry-blind collapse: a linear digit statistic invariant under every carry move has `w_{i+1} ≡ b·w_i` (`weight_rel_of_invariant`) and equals `w₀·N (mod m)` (`collapse`) | `NiceSearch.collapse_of_invariant` | — | prose proof | proved | 6 |
| THY-3 | once some output digits are fixed, the rest sum to the complement and form the complement set (`complement_set`): a digit-sum window on the unassigned positions is vacuous | `NiceSearch.complement_sum` | — | prose | proved | 6 |
| THY-4 | the `b²−1` sieve on `n² + n³` adds nothing to casting out `b−1`s when both powers have at least 3 digits (every nice band with `b ≥ 6`, RNG-3): every residue mod `b²−1` that the digit-sum congruence allows is `(x + y) mod (b²−1)` for digit words `x`, `y` of those lengths with nonzero leading digits that use every digit once (`sieve_b2_complete`; the `c`-subset sums of `0..b−1` fill an interval, `exists_subset_sum`). False without the length condition: base 4, lengths (2, 2) (Janzert's `base_four_sieve_is_incomplete`) | `NiceSearch.sieve_b2_complete` | — | brute force b 6–9, every length split; Janzert's `sieve_complete`, `pick_sum` | proved | 6 |
| THY-5 | middle-window filter is sound (digits at `p..p+w` depend on `n mod b^(p+w)`) | `NiceSearch.window_sound` | — | prose | proved | 6 |
| THY-6 | the interval-domain Hall check is strictly incomplete: base 10, `[47, 60]` has an SDR but no nice number (by `decide`) | `NiceSearch.hall_relaxation_incomplete` | — | probe | proved | 6 |
| THY-7 | carry-state ladder: distinct suffixes never share an exact future | `NiceSearch.Theory.suffix_future_injective` | — | measured | planned | 6 |
| THY-8 | tree recurrences for digits of n², n³ when appending a digit | `NiceSearch.Theory.tree_recurrence` | `scripts/radix_tree_search.rs` | 20k random cases | planned | 6 |
| THY-9 | witness model `λ_b = range_b · b!/b^b` (definition only) | `NiceSearch.witnessRate` | — | heuristic | def | 6 |
<!-- claims:end -->
