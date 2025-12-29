# Formal Verification of Nice Numbers

This directory contains formal proofs in Lean 4 for properties of square-cube pandigital ("nice") numbers.

## What are Nice Numbers?

A number `n` is **nice** in base `b` if the concatenation of the digits of `n²` and `n³` (when represented in base `b`) contains all digits `0` through `b-1` exactly once. This property is also called being "pandigital."

For example:
- `69` is nice in base 10: `69² = 4761` and `69³ = 328509`, giving digits `[4,7,6,1,3,2,8,5,0,9]` ✓
- `2` is nice in base 4: `2² = 4 = [1,0]₄` and `2³ = 8 = [2,0]₄`, giving digits `[1,0,2,0]` which simplifies to using digits `[0,1,2,3]` once each... (TODO: verify this example)

## Why Lean?

The computational search for nice numbers involves:
1. **Large search spaces** - checking trillions of candidates
2. **Complex filtering rules** - residue class constraints, digit count bounds
3. **Nonexistence claims** - proving certain bases have no nice numbers

Lean provides **machine-checked certainty** that our filtering rules are sound and our nonexistence proofs are correct. It does NOT help us find nice numbers faster - that's what the Rust/WASM code does.

## What We Prove

### Core Definitions (`nice.lean`)

1. **Pandigitality**: A list of digits is pandigital if it contains each digit `0..b-1` exactly once
2. **Niceness**: A number `n` is nice in base `b` if `digits(n²) ++ digits(n³)` is pandigital
3. **Digit sum and unique digit counting**

### Key Theorems

#### 1. Digit Count Necessity
```lean
theorem nice_digit_count {n b : ℕ} (h : Nice n b) :
    numDigits b (n^2) + numDigits b (n^3) = b
```
If `n` is nice in base `b`, then the total number of digits in `n²` and `n³` must equal `b`.

#### 2. Interval Bounds
```lean
theorem nice_interval_bounds {n b : ℕ} (h : Nice n b) :
    ∃ k ℓ : ℕ, k + ℓ + 2 = b ∧
               b^k ≤ n^2 ∧ n^2 < b^(k+1) ∧
               b^ℓ ≤ n^3 ∧ n^3 < b^(ℓ+1)
```
This constrains `n` to a specific interval based on `b`. The server uses these bounds to generate search ranges.

#### 3. Modular Constraints (Residue Filter)
```lean
theorem digit_sum_mod {b n : ℕ} (hb : b ≥ 2) :
    digitSumBase b n ≡ n [MOD b - 1]

theorem nice_mod_constraint {n b : ℕ} (h : Nice n b) :
    n^2 + n^3 ≡ b * (b - 1) / 2 [MOD b - 1]
```
The digit sum of any number `n` in base `b` is congruent to `n` modulo `b-1`. For nice numbers, the digit sum must be `b*(b-1)/2` (since we use all digits `0..b-1` exactly once). This gives us:

```
n² + n³ ≡ b*(b-1)/2 [MOD b-1]
```

This constraint eliminates most candidates! For example, in base 10, only residue classes `{0, 3, 6, 8}` mod 9 can possibly be nice.

#### 4. Filter Soundness
```lean
theorem residue_filter_sound {n b : ℕ} (hb : b ≥ 2) (h : Nice n b) :
    n % (b - 1) ∈ residueFilter b
```
The residue filter used in the computational search is provably sound.

#### 5. Nonexistence Results
```lean
theorem no_nice_when_one_mod_five {b : ℕ} (hb : b ≡ 1 [MOD 5]) :
    ∀ n, ¬Nice n b

theorem no_nice_when_filter_empty {b : ℕ} (hb : b ≥ 2) 
    (hemp : residueFilter b = ∅) : ∀ n, ¬Nice n b
```
Certain bases provably have no nice numbers due to digit count constraints or empty residue filters.

## Structure of the Proof

The proofs follow this dependency structure:

```
Basic Definitions
    ↓
Pandigital Properties (digit sum = b*(b-1)/2)
    ↓
Digit Count Theorem ← Interval Bounds
    ↓
Modular Constraints
    ↓
Filter Soundness → Nonexistence Results
```

## How This Integrates with the Codebase

The Rust/Python/WASM code performs the actual search:
```
1. Server generates search ranges (based on interval bound theorems)
2. Client filters candidates (using residue_filter - proved sound in Lean)
3. Client checks remaining candidates (using get_is_nice / get_num_unique_digits)
4. Results are verified via consensus
```

Lean guarantees that:
- The interval bounds are correct (no missed candidates)
- The residue filter is sound (no false negatives)
- Nonexistence claims for certain bases are rigorous

## Current Status

**Implemented:**
- ✅ Core definitions (Pandigital, Nice, digitSum, etc.)
- ✅ Theorem statements for all major results
- ⚠️  Most proofs are marked `sorry` (to be completed)

**Next Steps:**
1. Prove `pandigital_digit_sum` (the digit sum of a pandigital sequence)
2. Prove `digit_sum_mod` (digit sum congruence)
3. Prove `nice_digit_count` (digit count necessity)
4. Prove `nice_mod_constraint` (modular constraint)
5. Prove `residue_filter_sound` (filter soundness)
6. Complete nonexistence proofs

**Future Work:**
- Formalize the exact interval bounds for each `b ≡ r (mod 5)` case
- Verify specific nice numbers (69 in base 10, etc.)
- Formalize near-miss analysis and distribution statistics
- Prove bounds on the density of nice numbers (if they exist)

## Building and Running

### Prerequisites

1. Install [Lean 4](https://leanprover.github.io/lean4/doc/setup.html)
2. This project uses Lean 4.8.0 (specified in `lean-toolchain`)

### Setup

```bash
cd nice/proofs
lake update  # Fetch mathlib dependencies
lake build   # Build the project
```

### Checking Proofs

```bash
lake build Nice  # Check all proofs in nice.lean
```

### Interactive Development

Open `nice.lean` in VS Code with the Lean 4 extension to:
- See proof states interactively
- Get real-time feedback on definitions
- Use tactics like `rw`, `simp`, `ring`, etc.

## References

- [Original article on nice numbers](https://beautifulthorns.wixsite.com/home/post/is-69-unique)
- [Nice numbers search project](https://nicenumbers.net)
- [Lean 4 documentation](https://leanprover.github.io/lean4/doc/)
- [Mathlib4 documentation](https://leanprover-community.github.io/mathlib4_docs/)

## Contributing

To add new theorems or complete existing proofs:

1. Keep proofs modular and well-documented
2. Add examples that can be computationally verified
3. State theorems clearly before proving them
4. Use `sorry` for incomplete proofs, but document what needs to be done
5. When proving filter soundness, ensure the formalization matches the Rust implementation exactly

## License

This proof development is part of the Nice Numbers project. See the main LICENSE file in the repository root.
