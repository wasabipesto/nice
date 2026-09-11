# Design and phases

Self-contained summary of the formalization plan. `CLAIMS.md` is the
catalogue; `README.md` is the how-to.

## Why

The search has shipped two soundness bugs that no benchmark could catch,
because nice numbers are too rare for a wrongly-rejecting filter to show
up in any parity test: the cross MSD×LSD skip (2026-02 → v3.3.0, six
months, ~97 M submissions disqualified) and the v3.2.14 GPU prefilter
fallback that rejected everything on bases 10–25. Both were quantifier
errors ("for the range" vs "for each residue"; "digits exist" vs "digits
exist for this base"). Many load-bearing claims live only in doc comments,
some with stale or contradictory numbers. Formalization buys: every
filter's side conditions as explicit hypotheses discharged from the
base-range lemma; a certified constants table; one completeness theorem
for the niceonly cascade that any new filter must be slotted into; a
refutation channel for proposed filters; and a place where theorems can be
sharpened into new optimizations.

## Three layers and a bridge

- **Spec** (`Nice/Spec/`): the mathematics only. Positional digits
  `digit b n j = n / b^j % b` (least significant first, as
  `to_digits_asc`), `IsNice` as a `List.Perm`, the base range as a
  predicate.
- **Model** (`Nice/Model/`): computable Lean functions mirroring each Rust
  filter at the algorithmic level (tables, walks, recursion; not U256 limbs
  or magic-constant division), each with a soundness theorem against Spec
  ("if the model rejects, no nice number is lost") and, where cheap, an
  exactness theorem ("the model's table equals the spec's set").
- **Theory** (`Nice/Theory/`): what is true about the problem independent
  of the code; `Refuted.lean` for filters proved unsound with a witness;
  `Conjectures.lean` for proposals awaiting a proof or a counterexample.
- **Bridge**: `CLAIMS.md` (claim → Lean declaration → Rust site → status),
  `Lean:` tags in Rust doc comments checked by `scripts/check_claims.py`,
  and (from phase 2) fixtures emitted by Rust and diffed against the model.
  Theorems pin "model = spec"; fixtures pin "model = code".

Not in scope: verifying the Rust or CUDA implementations themselves,
performance claims, the witness-probability heuristic beyond its
definition. Compaction queue bounds and compiler quirks stay on device
tests and are listed in the registry as such.

Policy: no `native_decide`; `decide`/`norm_num` for concrete examples;
`proved` rows depend on no axiom beyond `propext`, `Classical.choice`,
`Quot.sound`. Proof debt (`stated`) is allowed and visible; a proved
theorem may not silently regress.

## Phases

| phase | content | exit |
|---|---|---|
| 0 | project, registry, tag checker, definitions, examples | `lake build` green; checker passing |
| 1 | base range exactness per `b mod 5`, `b ≡ 1 (mod 5)` empty, "≥ k digits" side condition, `Const.lean` numeric cutoffs | `Const.lean` sorry-free; constants tagged in Rust |
| 2 | residue, LSD, stride table walk, seeded check, GPU ordinal formula; first fixtures | "every nice n in `[s,e)` is a stride candidate" |
| 3 | MSD interval domains, width recurrence, Hall soundness, recursive subdivision cover, monotone rejection; Kuhn completeness last | subdivision cover theorem sorry-free |
| 4 | cross-end certificate with its `j ≥ k` guard, certificate inheritance, refutation of the 2026-02 skip, **END-1** | END-1 sorry-free; CI job wired; PR leaves draft |
| 5 | GPU tiling / lane partition / split16 / Horner / prefilter, field generators, detailed-mode lemmas | |
| 6 | residue-count closed form, carry-blind collapse, block-filter collapses, Hall-relaxation witness, tree recurrences | open-ended |

END-1: `∀ n ∈ range ⊆ baseRange b, IsNice b n → n ∈ Model.processRangeNiceonly b k range`,
and every reported number is nice. After it exists, "add a filter" means
"add a model function, prove its lemma, re-prove END-1 with it in the
chain".

## Workflow

Code → Lean: write the claim in `CLAIMS.md` with explicit hypotheses (if it
cannot be stated, that is the review finding); add the model function and
a fixture; prove or mark `stated`; tag the Rust site.

Lean → code: state a proposed filter in `Conjectures.lean` with a `decide`
check on bases 5–16; a failing check moves it to `Refuted.lean` with the
witness; a passing one earns proof effort, and the theorem's hypotheses are
the implementation's spec. Candidates already visible from the catalogue:
a per-(range, residue) Hall check with the residue's exact low digits
pre-assigned (strictly dominates the one-AND cross-end test); exact
prefilter digit tables replacing the float `ln` computation; the U256
cutoff has one base of slack (69 fits).
