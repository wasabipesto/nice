#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["tabulate"]
# ///
"""
Analyze filter effectiveness for a given base.
"""

import sqlite3
import sys
from pathlib import Path
from typing import List, Tuple

from tabulate import tabulate


def get_base_range(base: int) -> Tuple[int, int]:
    """Calculate the valid range for a base using the same formula as the Rust code."""
    k = base // 5
    b = base

    mod = base % 5

    if mod == 0:
        start = int(b ** (3 * k - 1) ** (1/3) + 0.5)  # ceiling_root
        end = b ** k
        return (start, end)
    elif mod == 1:
        return None
    elif mod == 2:
        start = b ** k
        end = int((b ** (3 * k + 1)) ** (1/3))  # floor_root
        return (start, end)
    elif mod == 3:
        start = int(b ** (3 * k + 1) ** (1/3) + 0.5)  # ceiling_root
        end = int((b ** (2 * k + 1)) ** 0.5)  # floor_root
        return (start, end)
    elif mod == 4:
        start = int((b ** (2 * k + 1)) ** 0.5 + 0.5)  # ceiling_root
        end = int((b ** (3 * k + 2)) ** (1/3))  # floor_root
        return (start, end)

    return None


def query_cached_ranges(db_path: str, base: int) -> List[Tuple[int, int, int]]:
    """Query all cached ranges for a base from the database.

    Returns list of (range_start, range_end, valid_size) tuples.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT range_start, range_end, valid_size
        FROM msd_cache
        WHERE base = ?
        ORDER BY range_start
    """, (base,))

    ranges = [(int(row[0]), int(row[1]), int(row[2])) for row in cursor.fetchall()]
    conn.close()

    return ranges


def calculate_chunk_filtering(
    base_start: int,
    base_end: int,
    cached_ranges: List[Tuple[int, int, int]],
    num_chunks: int
) -> List[Tuple[int, int, float]]:
    """Calculate the filtering percentage for each chunk of the base range.

    Returns list of (chunk_start, chunk_end, valid_pct) tuples.
    valid_pct: 0.0 = all filtered, 1.0 = none filtered
    """
    base_size = base_end - base_start
    chunk_size = base_size / num_chunks

    # Build a lookup structure for cached ranges
    range_map = {}
    for start, end, valid_size in cached_ranges:
        total_size = end - start
        if total_size > 0:
            valid_pct = valid_size / total_size
            range_map[(start, end)] = valid_pct

    chunks = []

    # For each chunk, find overlapping cached ranges and calculate filtering
    for i in range(num_chunks):
        chunk_start = base_start + int(i * chunk_size)
        chunk_end = base_start + int((i + 1) * chunk_size)
        if i == num_chunks - 1:
            chunk_end = base_end  # Last chunk takes remainder

        # Find the best cached data for this chunk
        best_valid_pct = 1.0  # Default to no filtering

        # Check for exact match or containing range
        for (r_start, r_end), valid_pct in range_map.items():
            if r_start <= chunk_start and r_end >= chunk_end:
                # This cached range contains our chunk
                best_valid_pct = valid_pct
                break

        # If no containing range, try to aggregate from subdivisions
        if best_valid_pct == 1.0:
            overlapping = []
            for (r_start, r_end), valid_pct in range_map.items():
                if r_end > chunk_start and r_start < chunk_end:
                    overlapping.append((r_start, r_end, valid_pct))

            if overlapping:
                # Calculate weighted average based on overlap
                total_overlap = 0
                weighted_sum = 0
                for r_start, r_end, valid_pct in overlapping:
                    overlap_start = max(chunk_start, r_start)
                    overlap_end = min(chunk_end, r_end)
                    overlap_size = overlap_end - overlap_start
                    if overlap_size > 0:
                        total_overlap += overlap_size
                        weighted_sum += overlap_size * valid_pct

                if total_overlap > 0:
                    best_valid_pct = weighted_sum / total_overlap

        chunks.append((chunk_start, chunk_end, best_valid_pct))

    return chunks


def analyze_effectiveness(db_path: str, base: int, num_chunks: int = 64):
    """Analyze filter effectiveness transition for a base."""
    # Get base range
    base_range = get_base_range(base)
    if base_range is None:
        print(f"Base {base} has no valid range (mod 5 = {base % 5})")
        return

    base_start, base_end = base_range
    total_size = base_end - base_start

    print(f"Base {base} range: {base_start:,} to {base_end:,} (total: {total_size:,})")

    # Query cached data
    print(f"Getting data from database {db_path}")
    cached_ranges = query_cached_ranges(db_path, base)
    if not cached_ranges:
        print(f"No cached data found for base {base}")
        return

    print(f"Found {len(cached_ranges)} cached ranges")
    print()

    # Calculate filtering for each chunk
    chunks = calculate_chunk_filtering(base_start, base_end, cached_ranges, num_chunks)

    # Prepare table data
    table_data = []
    for i, (chunk_start, chunk_end, valid_pct) in enumerate(chunks):
        filter_pct = 1.0 - valid_pct
        position = (i + 0.5) / num_chunks * 100  # Middle of chunk as % through range
        table_data.append([
            i,
            f"{chunk_start:,}",
            f"{chunk_end:,}",
            f"{valid_pct*100:.1f}%",
            f"{filter_pct*100:.1f}%",
            f"{position:.1f}%"
        ])

    # Print chunk analysis using tabulate
    print(f"Filter efficacy by chunk ({num_chunks} chunks):")
    print()
    headers = ["Chunk", "Start", "End", "Valid%", "Filter%", "Position"]
    print(tabulate(table_data, headers=headers, tablefmt="github"))
    print()


def main():
    if len(sys.argv) < 2:
        print("Usage: base_efficacy.py <base> [num_chunks]")
        print("Example: base_efficacy.py 50 64")
        sys.exit(1)

    try:
        base = int(sys.argv[1])
    except ValueError:
        print(f"Error: Base must be an integer, got '{sys.argv[1]}'")
        sys.exit(1)

    num_chunks = 64
    if len(sys.argv) >= 3:
        try:
            num_chunks = int(sys.argv[2])
        except ValueError:
            print(f"Error: num_chunks must be an integer, got '{sys.argv[2]}'")
            sys.exit(1)

    db_path = "cache/msd_cache.db"

    if not Path(db_path).exists():
        print(f"Error: Database not found at {db_path}")
        sys.exit(1)

    analyze_effectiveness(db_path, base, num_chunks=num_chunks)


if __name__ == '__main__':
    main()
