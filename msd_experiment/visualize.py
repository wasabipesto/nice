#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "matplotlib",
#     "numpy",
# ]
# ///
"""
Visualize MSD filter effectiveness as a heatmap.

This script reads cached MSD filter data from the SQLite database and generates
horizontal bar charts showing filtering effectiveness across the base range.

Each base is shown as a horizontal bar divided into chunks. The color indicates
filtering effectiveness:
- Red (0.0): Completely filtered out by MSD filter
- White (1.0): No filtering possible (must check all numbers)
- Gradient: Partial filtering

Usage:
    uv run visualize.py --base 40
    uv run visualize.py --base 40 --chunks 256 --output msd_viz.png
    uv run visualize.py --base-range 10 50 --output msd_all.png
"""

import argparse
import sqlite3
import sys
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np


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
) -> np.ndarray:
    """Calculate the filtering percentage for each chunk of the base range.

    Returns array of filtering percentages (0.0 = all filtered, 1.0 = none filtered).
    """
    base_size = base_end - base_start
    chunk_size = base_size / num_chunks

    filtering = np.ones(num_chunks)  # Start assuming no filtering (1.0)

    # Build a lookup structure for cached ranges
    # Map each cached range to a valid percentage
    range_map = {}
    for start, end, valid_size in cached_ranges:
        total_size = end - start
        if total_size > 0:
            valid_pct = valid_size / total_size
            range_map[(start, end)] = valid_pct

    # For each chunk, find overlapping cached ranges and calculate filtering
    for i in range(num_chunks):
        chunk_start = base_start + int(i * chunk_size)
        chunk_end = base_start + int((i + 1) * chunk_size)
        if i == num_chunks - 1:
            chunk_end = base_end  # Last chunk takes remainder

        # Find the best cached data for this chunk
        # Prefer exact matches, then containing ranges, then subdivisions
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

        filtering[i] = best_valid_pct

    return filtering


def visualize_base(
    db_path: str,
    base: int,
    num_chunks: int = 256,
    ax=None,
    show_title: bool = True
) -> None:
    """Visualize filtering effectiveness for a single base."""
    # Get base range
    base_range = get_base_range(base)
    if base_range is None:
        print(f"Base {base} has no valid range (mod 5 = 1)")
        return

    base_start, base_end = base_range

    # Query cached data
    cached_ranges = query_cached_ranges(db_path, base)
    if not cached_ranges:
        print(f"No cached data found for base {base}")
        return

    # Calculate filtering for each chunk
    filtering = calculate_chunk_filtering(base_start, base_end, cached_ranges, num_chunks)

    # Create colormap: red (filtered) to white (not filtered)
    colors = ['#cc0000', '#ff6666', '#ffcccc', '#ffffff']
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('filtering', colors, N=n_bins)

    # Create visualization
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 2))

    # Create the heatmap
    im = ax.imshow([filtering], aspect='auto', cmap=cmap, vmin=0, vmax=1,
                   extent=[0, num_chunks, 0, 1], interpolation='nearest')

    # Calculate overall statistics
    avg_filtering = 1 - filtering.mean()
    total_size = base_end - base_start

    # Set labels
    if show_title:
        ax.set_title(f'Base {base}: {total_size:.3e} numbers, {avg_filtering*100:.1f}% filtered',
                     fontsize=11, pad=10)
    else:
        ax.set_ylabel(f'Base {base}', fontsize=9)

    # ax.set_xlabel('Position in range', fontsize=9)
    ax.set_yticks([])
    ax.set_xticks([])
    # ax.set_xticks([0, num_chunks/4, num_chunks/2, 3*num_chunks/4, num_chunks])
    # ax.set_xticklabels(['Start', '25%', '50%', '75%', 'End'], fontsize=8)

    return im


def visualize_multiple_bases(
    db_path: str,
    base_start: int,
    base_end: int,
    num_chunks: int
) -> None:
    """Visualize filtering effectiveness for multiple bases in a single figure."""
    bases = []
    for base in range(base_start, base_end):
        if base % 5 != 1:  # Skip bases with no valid range
            base_range = get_base_range(base)
            if base_range is not None:
                bases.append(base)

    if not bases:
        print(f"No valid bases in range {base_start}-{base_end}")
        return

    # Create figure with subplots
    n_bases = len(bases)
    fig, axes = plt.subplots(n_bases, 1, figsize=(14, max(n_bases * 0.8, 8)))

    if n_bases == 1:
        axes = [axes]

    fig.suptitle('MSD Filter Effectiveness by Base', fontsize=14, y=0.995)

    # Plot each base
    im = None
    for i, base in enumerate(bases):
        im = visualize_base(db_path, base, num_chunks, axes[i], show_title=False)

    # Add colorbar
    if False:
        cbar = fig.colorbar(im, ax=axes, orientation='horizontal',
                           pad=0.05, aspect=40, shrink=0.8)
        cbar.set_label('Valid Percentage (White = Must Check, Red = Filtered Out)', fontsize=10)
        cbar.ax.tick_params(labelsize=8)

    plt.tight_layout()

    output = "output/msd_filter_effectiveness.png"
    plt.savefig(output, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {output}")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize MSD filter effectiveness from cached data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --base 40
  %(prog)s --base 40 --chunks 512 --output base40.png
  %(prog)s --base-range 10 50 --output all_bases.png
        """
    )

    parser.add_argument('--db-path', default='msd_cache.db',
                       help='Path to SQLite database (default: msd_cache.db)')
    parser.add_argument('--base', type=int,
                       help='Base to visualize')
    parser.add_argument('--base-range', type=int, nargs=2, metavar=('START', 'END'),
                       help='Range of bases to visualize (exclusive end)')
    parser.add_argument('--chunks', type=int, default=256,
                       help='Number of chunks to divide the range into (default: 256)')

    args = parser.parse_args()

    # Check if database exists
    if not Path(args.db_path).exists():
        print(f"Error: Database not found at {args.db_path}")
        sys.exit(1)

    # Validate arguments
    if args.base is None and args.base_range is None:
        parser.error("Must specify either --base or --base-range")

    if args.base is not None and args.base_range is not None:
        parser.error("Cannot specify both --base and --base-range")

    # Generate visualization
    if args.base is not None:
        fig, ax = plt.subplots(figsize=(14, 2))
        im = visualize_base(args.db_path, args.base, args.chunks, ax)
        plt.tight_layout()

        output = "output/msd_filter_effectiveness_single.png"
        plt.savefig(args.output, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {args.output}")
    else:
        visualize_multiple_bases(
            args.db_path,
            args.base_range[0],
            args.base_range[1],
            args.chunks
        )


if __name__ == '__main__':
    main()
