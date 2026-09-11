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
fixtures/       tables emitted by Rust, checked against the Lean model (phase 2)
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
| 1 | 0 | 12 | 0 | 6 | 0 |
| 2 | 0 | 9 | 0 | 1 | 0 |
| 3 | 0 | 0 | 0 | 9 | 0 |
| 4 | 0 | 0 | 0 | 5 | 0 |
| 5 | 0 | 0 | 0 | 11 | 0 |
| 6 | 0 | 0 | 0 | 9 | 0 |
| — | 0 | 0 | 0 | 0 | 2 |

Proved or defined so far:

- **DEF-1** `Nice.IsNice`: `IsNice b n` ⇔ the base-b digits of n² followed by those of n³ permute `0..b-1`
- **DEF-1a** `Nice.isNice_iff_pandigital`: the three-part `Pandigital` definition of `origin/proofs` is equivalent
- **RNG-1** `Nice.nice_digit_count`: `IsNice b n → numDigits(n²) + numDigits(n³) = b`
- **RNG-2** `Nice.memBaseRange_of_inBaseRange`: the per-`b mod 5` closed-form interval contains every n with `numDigits(n²) + numDigits(n³) = b`
- **RNG-3** `Nice.three_le_numDigits_of_inBaseRange`: inside the range every power has `≥ k` digits for `k ≤ 3`, `b ≥ 6`
- **RNG-4** `Nice.not_inBaseRange_of_one_mod_five`: `b ≡ 1 (mod 5)` ⇒ no n has digit-count sum b
- **RNG-5** `Nice.numDigits_pow_mono`: `numDigits b (n^e)` is monotone in n
- **NUM-1** `Nice.Const.u128_cutoff_40`: `(rangeEnd 40 − 1)^3 < 2^128`
- **NUM-2** `Nice.Const.u256_cutoff`: `(rangeEnd b − 1)^3 < 2^256` for `b ≤ 68`; 69 fits, 70 does not
- **NUM-3** `Nice.Const.max_fw_digits`: `numDigits b (n^3) ≤ 38` for `b ≤ 64` in range
- **NUM-4** `Nice.Const.stride_modulus_u32`: `(b−1)·b^3 < 2^32` for `b ≤ 256` (u32 stride table)
- **NUM-4a** `Nice.Const.stride_modulus_gpu`: `(b−1)·b^3 < 2^28` for `b ≤ 128` (`MAX_STRIDE_MODULUS`)
- **NUM-5** `Nice.Const.mask_width`: digit masks need `b ≤ 64` (u64) / `b ≤ 128` (two words)
- **NUM-8** `Nice.Const.mod_m_bound`: `M² + M < 2^64` for `M < 2^32`
- **NUM-9** `Nice.Const.histogram_bins`: histogram bins cannot overflow u32
- **RES-1** `Nice.mem_residueFilter_of_isNice`: `IsNice b n → n² + n³ ≡ b(b−1)/2 (mod b−1)`; `n mod (b−1) ∈ residueFilter b`
- **RES-1a** `Nice.nice_digit_sum`: a nice number's output digits sum to `b(b−1)/2`
- **RES-3** `Nice.no_nice_of_residueFilter_empty`: `residueFilter b = ∅ → ∀ n, ¬IsNice b n`; `residueFilter 11 = ∅`
- **LSD-1** `Nice.digit_pow_mod_pow`: `digit b (n^e) j` for `j < k` depends only on `n mod b^k`
- **LSD-2** `Nice.mem_lsdBitmap_of_isNice`: nice + RNG-3 ⇒ the 2k fixed-width low digits are pairwise distinct ⇒ `n mod b^k ∈ lsdBitmap b k`
- **STR-1** `Nice.mem_validResidues_iff`: `Coprime (b−1) (b^k)`; passes both ⇔ `n mod M ∈ validResidues`
- **STR-2** `Nice.walk_eq_filter`: the gap-table walk visits exactly the valid n in `[start,end)` in order
- **STR-3** `Nice.seeded_iff_isNice`: seeded check equals the plain check under RNG-3
- **STR-4** `Nice.lowMask_eq`: `low_digit_masks[i]` is exactly the low-digit set of residue i's powers
- **GPU-0** `Nice.exists_ordinal_eq`: ordinal formula `B0 + ⌊g/#V⌋·M + V[g mod #V]` is strictly increasing (`ordinal_strictMono`), always valid (`ordinal_valid`) and hits every valid n ≥ B0, so it enumerates the same set as STR-2
<!-- status:end -->
