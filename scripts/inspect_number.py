#!/usr/bin/env python3
"""
inspect_number.py

Inspect a number's "niceness" (square-cube pandigital) properties.

A number n is *nice* in base b if, when n² and n³ are written in base b,
their digit sequences together contain each digit 0 … b-1 exactly once.
Niceness = (unique digits in n² ∪ n³) / b.

This script:
  1. Finds every base for which n is a valid candidate — i.e. n falls in the
     digit-count window where n² and n³ have a combined total of exactly b
     digits in base b (necessary but not sufficient for niceness).
  2. For each such base, prints n², n³, their base-b representations, the
     combined digit sequence, niceness, position in the search range, and a
     full digit histogram.

Usage:
    python scripts/inspect_number.py 69
    python scripts/inspect_number.py 69 --max-base 20
    python scripts/inspect_number.py 3141592653589793 --base 50
    python scripts/inspect_number.py 100 --min-base 2 --max-base 30
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from typing import Optional


# ══════════════════════════════════════════════════════════════════════════════
# Exact integer-root helpers  (mirror malachite's CeilingRoot / FloorRoot)
# ══════════════════════════════════════════════════════════════════════════════

def _floor_sqrt(n: int) -> int:
    """Exact ⌊√n⌋ via Python's built-in arbitrary-precision isqrt."""
    return math.isqrt(n)


def _ceil_sqrt(n: int) -> int:
    """Exact ⌈√n⌉."""
    r = math.isqrt(n)
    return r if r * r == n else r + 1


def _floor_cbrt(n: int) -> int:
    """
    Exact ⌊∛n⌋ for arbitrary-precision integers.

    Uses Newton's method starting from a guaranteed overestimate
    (2^⌈bits(n)/3⌉), which keeps the iteration monotonically decreasing.
    """
    if n == 0:
        return 0
    # Guaranteed overestimate: 2^⌈log₂(n)/3⌉ ≥ ∛n
    bits = n.bit_length()
    x = 1 << ((bits + 2) // 3)
    while True:
        x2  = x * x
        x1  = (2 * x + n // x2) // 3   # Newton step: (2x + n/x²) / 3
        if x1 >= x:
            break
        x = x1
    # Final ±1 sanity check (should rarely fire)
    while x * x * x > n:
        x -= 1
    while (x + 1) ** 3 <= n:
        x += 1
    return x


def _ceil_cbrt(n: int) -> int:
    """Exact ⌈∛n⌉."""
    r = _floor_cbrt(n)
    return r if r * r * r == n else r + 1


# ══════════════════════════════════════════════════════════════════════════════
# Valid candidate range  (mirrors common/src/base_range.rs exactly)
# ══════════════════════════════════════════════════════════════════════════════

def get_base_range(base: int) -> Optional[tuple[int, int]]:
    """
    Return the half-open range [start, end) of valid candidates n for *base*,
    or None if no such range exists.

    Derivation (k = base // 5, following base_range.rs):
      b ≡ 0 (mod 5)  →  [ ⌈∛(b^(3k−1))⌉,   b^k                 )
      b ≡ 1 (mod 5)  →  (no valid range)
      b ≡ 2 (mod 5)  →  [ b^k,               ⌊∛(b^(3k+1))⌋      )
      b ≡ 3 (mod 5)  →  [ ⌈∛(b^(3k+1))⌉,   ⌊√(b^(2k+1))⌋       )
      b ≡ 4 (mod 5)  →  [ ⌈√(b^(2k+1))⌉,   ⌊∛(b^(3k+2))⌋      )
    """
    k   = base // 5
    mod = base % 5

    if mod == 0:
        if k == 0:
            return None                          # base=0 is degenerate
        start = _ceil_cbrt(base ** (3 * k - 1))
        end   = base ** k
    elif mod == 1:
        return None
    elif mod == 2:
        start = base ** k
        end   = _floor_cbrt(base ** (3 * k + 1))
    elif mod == 3:
        start = _ceil_cbrt(base ** (3 * k + 1))
        end   = _floor_sqrt(base ** (2 * k + 1))
    else:                                        # mod == 4
        start = _ceil_sqrt(base ** (2 * k + 1))
        end   = _floor_cbrt(base ** (3 * k + 2))

    return None if start >= end else (start, end)


# ══════════════════════════════════════════════════════════════════════════════
# Digit utilities
# ══════════════════════════════════════════════════════════════════════════════

def _to_digits_lsf(n: int, base: int) -> list[int]:
    """Digits of n in *base*, least-significant first.  Returns [0] for n = 0."""
    if n == 0:
        return [0]
    digits: list[int] = []
    while n:
        digits.append(n % base)
        n //= base
    return digits


def _digit_char(d: int) -> str:
    """0–9 → '0'–'9',  10–35 → 'a'–'z',  36+ → '[N]'."""
    if d < 10:
        return str(d)
    if d < 36:
        return chr(ord("a") + d - 10)
    return f"[{d}]"


def _fmt_number(n: int, base: int) -> str:
    """Format *n* in the given *base* as a human-readable string."""
    if n == 0:
        return "0"
    digits = _to_digits_lsf(n, base)
    if base > 36:
        return "·".join(f"[{d}]" for d in reversed(digits))
    return "".join(_digit_char(d) for d in reversed(digits))


def _fmt_digit_list(digits_lsf: list[int], base: int) -> str:
    """Format a digit list (LSF order) back to a readable string."""
    if base > 36:
        return "·".join(f"[{d}]" for d in reversed(digits_lsf))
    return "".join(_digit_char(d) for d in reversed(digits_lsf))


def _digit_set_str(digits: list[int]) -> str:
    """Format a sorted list of digit values as a set-style string."""
    if not digits:
        return "(none)"
    return "{" + ", ".join(_digit_char(d) for d in digits) + "}"


def _ordinal(n: int) -> str:
    """1 → '1st',  2 → '2nd',  11 → '11th', …"""
    if 11 <= n % 100 <= 13:
        return f"{n}th"
    return f"{n}" + {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")


# ══════════════════════════════════════════════════════════════════════════════
# Niceness computation  (mirrors get_num_unique_digits / get_is_nice)
# ══════════════════════════════════════════════════════════════════════════════

def compute_niceness(n: int, base: int) -> dict:
    """
    Compute all niceness metrics for *n* in *base*.

    Returned dict keys
    ------------------
    sq, cu              n² and n³
    sq_digits, cu_digits  digit lists (LSF) of n², n³ in base b
    sq_len, cu_len, total_len  digit counts
    counts              Counter of every digit across n² ++ n³
    unique_digits       sorted list of digits that appear at least once
    num_uniques         len(unique_digits)
    missing_digits      digits in 0…b-1 that never appear
    repeated_digits     digits that appear more than once
    niceness            num_uniques / base
    is_nice             True iff num_uniques == base  (100% pandigital)
    is_saved            True iff num_uniques > floor(0.9 * base)
                        (the threshold used by the search system to save a number)
    """
    sq = n * n
    cu = sq * n

    sq_d = _to_digits_lsf(sq, base)
    cu_d = _to_digits_lsf(cu, base)

    counts        = Counter(sq_d + cu_d)
    unique_digits = sorted(counts)
    num_uniques   = len(unique_digits)

    return dict(
        sq=sq,
        cu=cu,
        sq_digits=sq_d,
        cu_digits=cu_d,
        sq_len=len(sq_d),
        cu_len=len(cu_d),
        total_len=len(sq_d) + len(cu_d),
        counts=counts,
        unique_digits=unique_digits,
        num_uniques=num_uniques,
        missing_digits=sorted(d for d in range(base) if d not in counts),
        repeated_digits=sorted(d for d, c in counts.items() if c > 1),
        niceness=num_uniques / base,
        is_nice=num_uniques == base,
        is_saved=num_uniques > math.floor(0.9 * base),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Display
# ══════════════════════════════════════════════════════════════════════════════

_W = 66   # display width


def _hbar(ch: str = "═") -> str:
    return ch * _W


def _kv(key: str, value: str, key_w: int = 28, indent: int = 2) -> None:
    """Print a key-value row, left-padding the key to *key_w* characters."""
    print(f"{' ' * indent}{key:<{key_w}} {value}")


def _print_base_block(n: int, base: int, r_start: int, r_end: int) -> None:
    """Print the complete analysis block for one base."""
    r_size   = r_end - r_start
    position = n - r_start + 1            # 1-indexed position inside the range
    pct      = position / r_size * 100.0

    m = compute_niceness(n, base)

    # ── section heading ───────────────────────────────────────────────────────
    if m["is_nice"]:
        marker = "★  BASE"
    elif m["is_saved"]:
        marker = "✦  BASE"
    else:
        marker = "   BASE"
    print(_hbar("─"))
    print(f"  {marker} {base}   range [{r_start:,}, {r_end:,})")
    print(_hbar("─"))
    print()

    # ── base-b representations ────────────────────────────────────────────────
    n_digits_in_base = len(_to_digits_lsf(n, base))
    _kv(f"n   (base {base})",  _fmt_number(n,      base))
    _kv(f"n²  (base {base})", _fmt_number(m["sq"], base))
    _kv(f"n³  (base {base})", _fmt_number(m["cu"], base))
    print()

    # ── combined digit sequence ───────────────────────────────────────────────
    # Show n² digits first (MSB→LSB), then ‖, then n³ digits.
    sq_part = _fmt_digit_list(m["sq_digits"], base)
    cu_part = _fmt_digit_list(m["cu_digits"], base)
    if base > 36:
        # Separate lines are more readable for wide bracket notation
        _kv("n² digit sequence", sq_part)
        _kv("n³ digit sequence", cu_part)
    else:
        _kv("Combined  (n²  ‖  n³)", f"{sq_part}  ‖  {cu_part}")
    print()

    # ── digit counts ──────────────────────────────────────────────────────────
    total_ok = "✓" if m["total_len"] == base else "✗"
    _kv("n digit count",          str(n_digits_in_base))
    _kv("n² digit count",         str(m["sq_len"]))
    _kv("n³ digit count",         str(m["cu_len"]))
    _kv("Total  (n² + n³)",
        f"{m['total_len']}   (must equal {base} for candidates  {total_ok})")
    print()

    # ── unique / missing / repeated ───────────────────────────────────────────
    _kv("Unique digits",
        f"{m['num_uniques']} / {base}   →   {_digit_set_str(m['unique_digits'])}")
    _kv("Missing digits",  _digit_set_str(m["missing_digits"]))
    _kv("Repeated digits", _digit_set_str(m["repeated_digits"]))
    print()

    # ── niceness ─────────────────────────────────────────────────────────────
    near_miss_threshold = math.floor(0.9 * base)
    ratio_str = (f"{m['num_uniques']}/{base}"
                 f" = {m['niceness']:.6f}"
                 f"  ({m['niceness'] * 100:.4f}%)")
    if m["is_nice"]:
        ratio_str += "   ← ★ 100% NICE!"
    elif m["is_saved"]:
        ratio_str += "   ← ✦ near-miss (saved by search system)"
    _kv("Niceness", ratio_str)
    _kv("Near-miss threshold",
        f"num_uniques > {near_miss_threshold}"
        f"   (= floor(0.9 × {base}),"
        f" ≈ {near_miss_threshold / base:.4f} niceness)")
    print()

    # ── position in range ────────────────────────────────────────────────────
    _kv("Range size",   f"{r_size:,} candidates")
    _kv("n's position", f"{_ordinal(position)} of {r_size:,}   ({pct:.4f}% through range)")
    print()

    # ── digit histogram  (shown for base ≤ 50; too wide above that) ──────────
    if base <= 50:
        print("  Digit histogram  (n² and n³ combined):")
        for d in range(base):
            c    = m["counts"].get(d, 0)
            bar  = "█" * c
            if c == 0:
                note = "  ✗  missing"
            elif c == 1:
                note = "  ✓"
            else:
                note = f"  ✗  duplicate (×{c})"
            print(f"    {_digit_char(d):>4} : {bar:<6}{note}")
        print()


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(
        prog="inspect_number.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "number",
        help="Integer to inspect (arbitrarily large values are supported)",
    )
    ap.add_argument(
        "--min-base", "-m",
        type=int, default=2, metavar="B",
        help="Minimum base to check (default: 2)",
    )
    ap.add_argument(
        "--max-base", "-M",
        type=int, default=200, metavar="B",
        help="Maximum base to check (default: 200)",
    )
    ap.add_argument(
        "--base", "-b",
        type=int, default=None, metavar="B",
        help="Check one specific base only (overrides --min-base / --max-base)",
    )

    args = ap.parse_args()

    # ── parse input ───────────────────────────────────────────────────────────
    try:
        n = int(args.number)
    except ValueError:
        sys.exit(f"error: '{args.number}' is not a valid integer")
    if n < 0:
        sys.exit("error: number must be non-negative")

    if args.base is not None:
        min_b, max_b = args.base, args.base
    else:
        min_b, max_b = args.min_base, args.max_base
    if min_b < 2:
        sys.exit("error: minimum base must be ≥ 2")

    # ── header ────────────────────────────────────────────────────────────────
    print()
    print(_hbar("═"))
    print(f"  Inspecting:  {n:,}")
    print(_hbar("═"))
    print()

    # ── decimal summary ───────────────────────────────────────────────────────
    sq = n * n
    cu = sq * n
    _kv("n   (decimal)", f"{n:,}")
    _kv("n²  (decimal)", f"{sq:,}")
    _kv("n³  (decimal)", f"{cu:,}")
    print()

    # ── find every valid base ─────────────────────────────────────────────────
    valid: list[tuple[int, int, int]] = []
    for b in range(min_b, max_b + 1):
        rng = get_base_range(b)
        if rng and rng[0] <= n < rng[1]:
            valid.append((b, rng[0], rng[1]))

    span = (f"base {min_b}" if min_b == max_b
            else f"bases {min_b}–{max_b}")

    if not valid:
        print(f"  ✗  Not a valid candidate in any searched {span}.")
        print()
        print("  Recall: bases where b ≡ 1 (mod 5) have no valid range by definition.")
        print("  For all other bases, n must lie inside the digit-count window where")
        print("  n² and n³ together have exactly b digits in base b.")
        return

    label = "1 base" if len(valid) == 1 else f"{len(valid)} bases"
    print(f"  ✓  Valid candidate in {label}  (searched {span}).")
    print()

    # ── per-base analysis ─────────────────────────────────────────────────────
    for b, start, end in valid:
        _print_base_block(n, b, start, end)


if __name__ == "__main__":
    main()
