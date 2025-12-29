# Formal Verification of Nice Numbers

This directory contains formal proofs in Lean 4 for properties of square-cube pandigital ("nice") numbers.

## How This Integrates with the Codebase

The Rust/Python/WASM code performs the actual search:
```
1. Server generates search ranges
2. Client filters candidates
3. Client checks remaining candidates
4. Results are verified via consensus
```

Lean guarantees that:
- The interval bounds are correct (no missed candidates)
- The residue filter is sound (no false negatives)
- Nonexistence claims for certain bases are rigorous

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
lake build   # Build the project
```

## References

- [Nice numbers search project](https://nicenumbers.net)
- [Original article on nice numbers](https://beautifulthorns.wixsite.com/home/post/is-69-unique)
- [Progress update on the search for nice numbers](https://beautifulthorns.wixsite.com/home/post/progress-update-on-the-search-for-nice-numbers)
- [Lean 4 documentation](https://leanprover.github.io/lean4/doc/)
- [Mathlib4 documentation](https://leanprover-community.github.io/mathlib4_docs/)
