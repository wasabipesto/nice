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
| DEF-1 | `IsNice b n` ⇔ the base-b digits of n² followed by those of n³ permute `0..b-1` | `Nice.IsNice` | `common/src/client_process.rs::get_is_nice` | tests | def | 0 |
| DEF-1a | the three-part `Pandigital` definition of `origin/proofs` is equivalent | `Nice.isNice_iff_pandigital` | — | — | proved | 0 |
| DEF-2 | inside the base range, `numUniques b n = b ↔ IsNice b n` | `Nice.numUniques_eq_iff_isNice` | `common/src/client_process.rs::get_num_unique_digits` | comment | planned | 1 |
| DEF-3 | `1 ≤ numUniques b n` for `n ≥ 1` (histogram bin 0 is empty) | `Nice.one_le_numUniques` | `common/src/distribution_stats.rs` | comment | planned | 1 |
| DEF-4 | near-miss cutoff is `⌊0.9 b⌋`, strict `>` | `Nice.nearMissCutoff` | `common/src/number_stats.rs::get_near_miss_cutoff` | tests | planned | 1 |
| RNG-1 | `IsNice b n → numDigits(n²) + numDigits(n³) = b` | `Nice.nice_digit_count` | `common/src/base_range.rs` | tests | proved | 0 |
| RNG-2 | the per-`b mod 5` closed-form interval contains every n with `numDigits(n²) + numDigits(n³) = b` | `Nice.memBaseRange_of_inBaseRange` | `common/src/base_range.rs::get_base_range_natural` | tests pin 8 bases | proved | 1 |
| RNG-2b | conversely every n in the closed-form interval has digit-count sum b (`InBaseRange`) | `Nice.inBaseRange_of_memBaseRange` | `common/src/base_range.rs::get_base_range_natural` | tests pin 8 bases | planned | 1 |
| RNG-3 | inside the range every power has `≥ k` digits for `k ≤ 3`, `b ≥ 6` | `Nice.three_le_numDigits_of_inBaseRange` | `common/src/lsd_filter.rs`, `common/src/client_process.rs::get_is_nice_with_known_lsd` | comment | proved | 1 |
| RNG-4 | `b ≡ 1 (mod 5)` ⇒ no n has digit-count sum b | `Nice.not_inBaseRange_of_one_mod_five` | `common/src/base_range.rs` | asserted | proved | 1 |
| RNG-5 | `numDigits b (n^e)` is monotone in n | `Nice.numDigits_pow_mono` | `common/src/gpu_config.rs::prefilter_params` | comment | proved | 1 |
| NUM-1 | `(rangeEnd 40 − 1)^3 < 2^128` | `Nice.Const.u128_cutoff_40` | `common/src/client_process.rs::MAX_BASE_FOR_FIXED_WIDTH_U128` | rust test (PR #158) | proved | 1 |
| NUM-2 | `(rangeEnd b − 1)^3 < 2^256` for `b ≤ 68`; 69 fits, 70 does not | `Nice.Const.u256_cutoff` | `common/src/client_process.rs::MAX_BASE_FOR_FIXED_WIDTH_U256` | rust test (PR #158) | proved | 1 |
| NUM-3 | `numDigits b (n^3) ≤ 38` for `b ≤ 64` in range | `Nice.Const.max_fw_digits` | `common/src/msd_prefix_filter.rs::MAX_FW_DIGITS` | comment | proved | 1 |
| NUM-4 | `(b−1)·b^3 < 2^32` for `b ≤ 256` (u32 stride table) | `Nice.Const.stride_modulus_u32` | `common/src/stride_filter.rs::StrideTable::new` | runtime assert | proved | 1 |
| NUM-4a | `(b−1)·b^3 < 2^28` for `b ≤ 128` (`MAX_STRIDE_MODULUS`) | `Nice.Const.stride_modulus_gpu` | `common/src/gpu_niceonly.rs::MAX_STRIDE_MODULUS` | runtime assert | proved | 1 |
| NUM-5 | digit masks need `b ≤ 64` (u64) / `b ≤ 128` (two words) | `Nice.Const.mask_width` | `common/src/stride_filter.rs::StrideTable::low_digit_masks` | const-assert | proved | 1 |
| NUM-6 | GPU chunk constants maximal and split16-safe | `Nice.Const.chunk_constants` | `common/src/gpu_config.rs::chunk_constants` | tests | planned | 1 |
| NUM-7 | per-base prefilter digit count equals the exact integer bound | `Nice.Const.prefilter_digits` | `common/src/gpu_config.rs::prefilter_params` | integer test re-checks the float | planned | 1 |
| NUM-8 | `M² + M < 2^64` for `M < 2^32` | `Nice.Const.mod_m_bound` | `common/src/cuda/nice_kernels.cu::mod_m` | comment (fixed PR #158) | proved | 1 |
| NUM-9 | histogram bins cannot overflow u32 | `Nice.Const.histogram_bins` | `common/src/cubecl_backend.rs::DRAIN_INTERVAL` | const-assert + test | proved | 1 |
| RES-1 | `IsNice b n → n² + n³ ≡ b(b−1)/2 (mod b−1)`; `n mod (b−1) ∈ residueFilter b` | `Nice.mem_residueFilter_of_isNice` | `common/src/residue_filter.rs::get_residue_filter` | tests | proved | 2 |
| RES-1a | a nice number's output digits sum to `b(b−1)/2` | `Nice.nice_digit_sum` | `common/src/residue_filter.rs` | — | proved | 0 |
| RES-2 | `b ≡ 3 (mod 4) → ∀ n, ¬IsNice b n` | `Nice.no_nice_of_three_mod_four` | `common/src/residue_filter.rs` (oracle test) | oracle test 5–512 | planned | 2 |
| RES-3 | `residueFilter b = ∅ → ∀ n, ¬IsNice b n`; `residueFilter 11 = ∅` | `Nice.no_nice_of_residueFilter_empty` | `common/src/gpu_niceonly.rs::residue_empty_result` | tests | proved | 2 |
| LSD-1 | `digit b (n^e) j` for `j < k` depends only on `n mod b^k` | `Nice.digit_pow_mod_pow` | `common/src/lsd_filter.rs` | comment | proved | 2 |
| LSD-2 | nice + RNG-3 ⇒ the 2k fixed-width low digits are pairwise distinct ⇒ `n mod b^k ∈ lsdBitmap b k` | `Nice.mem_lsdBitmap_of_isNice` | `common/src/lsd_filter.rs::get_valid_multi_lsd_bitmap` | brute-force b 4–16 | proved | 2 |
| STR-1 | `Coprime (b−1) (b^k)`; passes both ⇔ `n mod M ∈ validResidues` | `Nice.mem_validResidues_iff` | `common/src/stride_filter.rs::StrideTable::new` | comment | proved | 2 |
| STR-2 | the gap-table walk visits exactly the valid n in `[start,end)` in order | `Nice.walk_eq_filter` | `common/src/stride_filter.rs::iterate_range_masked` | tests (gap sum = M) | proved | 2 |
| STR-3 | seeded check equals the plain check under RNG-3 | `Nice.seeded_iff_isNice` | `common/src/client_process.rs::get_is_nice_with_known_lsd` | 5000-sample test | proved | 2 |
| STR-4 | `low_digit_masks[i]` is exactly the low-digit set of residue i's powers | `Nice.lowMask_eq` | `common/src/stride_filter.rs::StrideTable::new` | by construction | proved | 2 |
| GPU-0 | ordinal formula `B0 + ⌊g/#V⌋·M + V[g mod #V]` is strictly increasing (`ordinal_strictMono`), always valid (`ordinal_valid`) and hits every valid n ≥ B0, so it enumerates the same set as STR-2 | `Nice.exists_ordinal_eq` | `common/src/cuda/nice_kernels.cu`, `common/src/vulkan/codegen.rs`, `common/src/cubecl_backend.rs` | host-mirror tests | proved | 2 |
| MSD-1 | interval digit domains are a superset of the digits that occur | `Nice.Model.Msd.digit_mem_domain` | `common/src/msd_prefix_filter.rs::collect_power_domains` | tests | planned | 3 |
| MSD-2 | width recurrence exact; `diff ≥ b−1` ⇒ every lower position full | `Nice.Model.Msd.diff_recurrence` | `common/src/msd_prefix_filter.rs::collect_power_domains` | comment | planned | 3 |
| MSD-3 | dropping a power's domains (unequal digit counts) is sound | `Nice.Model.Msd.drop_power_sound` | `common/src/msd_prefix_filter.rs::analyze_msd_prefix` | comment | planned | 3 |
| MSD-4 | Hall soundness: no injective digit choice ⇒ no nice n in the range | `Nice.Model.Msd.rejected_sound` | `common/src/msd_prefix_filter.rs::has_distinct_assignment` | brute-force b 4–16 | planned | 3 |
| MSD-5 | Kuhn completeness: the augmenting-path search returns false only when no SDR exists | `Nice.Model.Hall.kuhn_complete` | `common/src/msd_prefix_filter.rs::hall_augment` | oracle test (PR #157) | planned | 3 |
| MSD-6 | domain-slot overflow only drops constraints | `Nice.Model.Msd.slot_overflow_sound` | `common/src/msd_prefix_filter.rs::collect_power_domains` | comment | planned | 3 |
| MSD-7 | recursive subdivision: leaves ⊇ every nice n; leaves disjoint | `Nice.Model.Msd.leaves_cover` | `common/src/msd_prefix_filter.rs::get_valid_ranges_recursive_masked` | brute-force b 4–16 | planned | 3 |
| MSD-8 | the over-64 prefix path is the singleton-domain case of MSD-4 | `Nice.Model.Msd.prefix_path_sound` | `common/src/msd_prefix_filter.rs::analyze_range_over_64` | comment | planned | 3 |
| MSD-9 | monotone rejection: `Rejected(I) ∧ J ⊆ I → Rejected(J)`; ancestor masks ⊆ leaf's own | `Nice.Model.Msd.rejected_mono` | `common/src/gpu_niceonly.rs::BlockTiling` | one-window test | planned | 3 |
| CRS-1 | singleton high digit at position `≥ k` colliding with a residue's exact low digit kills the residue in the range | `Nice.Model.Cross.certificate_sound` | `common/src/msd_prefix_filter.rs::analyze_range`, `common/src/stride_filter.rs::iterate_range_masked` | tests (b10/17/22/25) | planned | 4 |
| CRS-2 | a certificate for a range holds on every sub-range | `Nice.Model.Cross.certificate_mono` | `common/src/msd_prefix_filter.rs::get_valid_ranges_recursive_masked` | identical-leaves test | planned | 4 |
| CRS-3 | empty certificate for `b > 64` and size-1 ranges is the k=0 instance | `Nice.Model.Cross.certificate_empty` | `common/src/vulkan/mod.rs` | comment | planned | 4 |
| REF-1 | the removed MSD×LSD skip is unsound: witness b=10, k=2, `[68,70)` | `Nice.Refuted.msd_lsd_skip` | `common/src/msd_prefix_filter.rs` (NOTE) | regression test | planned | 4 |
| END-1 | the modelled niceonly pipeline reports every nice n in a range inside the base range, and only nice ones | `Nice.Model.NiceOnly.complete` | `common/src/client_process.rs::process_range_niceonly` | small-base brute force | planned | 4 |
| GPU-1 | block tiling partitions the field | `Nice.Model.Gpu.tiling_partition` | `common/src/gpu_niceonly.rs::BlockTiling::new` | test | planned | 5 |
| GPU-2 | block starts yield the same leaves and masks as chunk starts | `Nice.Model.Gpu.block_start_eq` | `common/src/gpu_niceonly.rs` | one test | planned | 5 |
| GPU-3 | mixing floors within a field loses nothing | `Nice.Model.Gpu.floor_mix_sound` | `common/src/gpu_niceonly.rs` | comment | planned | 5 |
| GPU-4 | lane tiling partitions the ordinals for any lane count | `Nice.Model.Gpu.lane_partition` | `common/src/gpu_niceonly.rs::lane_shift_for` | device tests | planned | 5 |
| GPU-5 | split16 chunk step is exact when `d < 2^16` | `Nice.Model.Gpu.split16_exact` | `common/src/vulkan/codegen.rs` | test | planned | 5 |
| GPU-6 | truncated schoolbook multiply ≡ reduction mod `d^limbs`; no u32 overflow | `Nice.Model.Gpu.truncated_mul` | `common/src/vulkan/codegen.rs`, `common/src/gpu_config.rs` | tests | planned | 5 |
| GPU-7 | chunked Horner keeps `acc << c` in u32 and covers the offset | `Nice.Model.Gpu.horner_exact` | `common/src/gpu_niceonly.rs::stride_chunk_bits` | tests | planned | 5 |
| GPU-8 | prefilter = LSD-2 at depth p plus NUM-7; without NUM-7 it rejects everything | `Nice.Model.Gpu.prefilter_sound` | `common/src/gpu_config.rs::prefilter_params` | tests | planned | 5 |
| GPU-9 | `mod_m` via `2^64 mod M` is correct under NUM-8 | `Nice.Model.Gpu.mod_m_correct` | `common/src/cuda/nice_kernels.cu::mod_m` | test | planned | 5 |
| GPU-C | warp/cube compaction queue bounds and uniformity | — | `common/src/cuda/nice_kernels.cu`, `common/src/cubecl_backend.rs` | device tests | device-test | — |
| FLD-1 | field and chunk generators partition the base; start-point chunk matching is correct | `Nice.Model.Fields.partition` | `common/src/generate_fields.rs`, `common/src/generate_chunks.rs`, `common/src/db_util/chunks.rs::reassign_fields_to_chunks` | tests | planned | 5 |
| FLD-DB | the stored field/chunk rows partition each base | — | `common/src/db_util/audit.rs` (PR #159) | sql audit | sql-audit | — |
| DET-1 | accumulator equals single pass; top-N of a superset contains the top-N | `Nice.Model.Detailed.accumulate_eq` | `common/src/distribution_stats.rs`, `common/src/number_stats.rs` | tests | planned | 5 |
| THY-1 | residue-count closed form over the prime powers of `b−1` | `Nice.Theory.residueFilter_card` | `common/src/residue_filter.rs` (oracle test) | oracle test 5–512 | planned | 6 |
| THY-2 | carry-blind collapse: a carry-invariant linear digit statistic is `w₀·N (mod m)` | `Nice.Theory.collapse` | — | prose proof | planned | 6 |
| THY-3 | digit-sum windows in a MITM join are vacuous | `Nice.Theory.window_vacuous` | — | prose | planned | 6 |
| THY-4 | `b²−1` block filter collapses (r-subset sums fill an interval of length ≥ b) | `Nice.Theory.subsetSum_interval` | — | prose proof | planned | 6 |
| THY-5 | middle-window filter is sound (digits at `p..p+w` depend on `n mod b^(p+w)`) | `Nice.Theory.window_sound` | — | prose | planned | 6 |
| THY-6 | Hall's marginal relaxation is strictly incomplete (witness) | `Nice.Theory.hall_relaxation_incomplete` | — | probe | planned | 6 |
| THY-7 | carry-state ladder: distinct suffixes never share an exact future | `Nice.Theory.suffix_future_injective` | — | measured | planned | 6 |
| THY-8 | tree recurrences for digits of n², n³ when appending a digit | `Nice.Theory.tree_recurrence` | `scripts/radix_tree_search.rs` | 20k random cases | planned | 6 |
| THY-9 | witness model `λ_b = range_b · b!/b^b` (definition only) | `Nice.Theory.witnessRate` | — | heuristic | planned | 6 |
<!-- claims:end -->
