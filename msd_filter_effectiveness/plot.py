# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "matplotlib",
#     "numpy",
#     "tqdm",
# ]
# ///

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def main():
    # Load aggregated stats
    stats_path = Path("output/aggregated_stats.json")
    with open(stats_path) as f:
        aggregated_stats = json.load(f)

    print(f"Loaded aggregated stats for {len(aggregated_stats)} bases")

    # Convert 1000 chunks to 100 chunks for display
    num_chunks_display = 100
    chunks_per_display = 10  # 1000 / 100

    chunk_effectiveness = {}

    for base_str, chunks in aggregated_stats.items():
        base = int(base_str)
        means = []

        # Process in groups of 10 to average down to 100 chunks
        for i in range(num_chunks_display):
            start_idx = i * chunks_per_display
            end_idx = start_idx + chunks_per_display

            # Aggregate the 10 chunks into one
            total_sum = 0.0
            total_count = 0

            for chunk in chunks[start_idx:end_idx]:
                total_sum += chunk["sum"]
                total_count += chunk["count"]

            if total_count > 0:
                means.append(total_sum / total_count)
            else:
                means.append(np.nan)

        chunk_effectiveness[base] = means

    print("Aggregated to 100 chunks for display")

    # Plot filter effectiveness per chunk for all bases
    fig, ax = plt.subplots(figsize=(14, 8))

    # Get sorted bases
    bases = sorted(chunk_effectiveness.keys())

    # Create a color map
    colors = plt.cm.viridis(np.linspace(0, 1, len(bases)))

    for idx, base in enumerate(bases):
        means = chunk_effectiveness[base]
        x = np.arange(num_chunks_display)

        # Plot only non-NaN values
        valid_mask = ~np.isnan(means)
        if np.any(valid_mask):
            ax.plot(
                x[valid_mask],
                np.array(means)[valid_mask],
                label=f"Base {base}",
                color=colors[idx],
                alpha=0.7,
                linewidth=1.5
            )

    ax.set_xlabel("Chunk Index (0-99)", fontsize=12)
    ax.set_ylabel("Mean Filter Effectiveness", fontsize=12)
    ax.set_title("MSD Filter Effectiveness by Base and Position", fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, num_chunks_display - 1)
    ax.set_ylim(0, 1)

    plt.tight_layout()

    # Save the plot
    output_path = Path("output/msd_filter_effectiveness_plot.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {output_path}")

    # Also create a heatmap view
    fig2, ax2 = plt.subplots(figsize=(14, 10))

    # Prepare data for heatmap
    heatmap_data = []
    bases_sorted = sorted(chunk_effectiveness.keys())
    for base in bases_sorted:
        heatmap_data.append(chunk_effectiveness[base])

    heatmap_array = np.array(heatmap_data)

    im = ax2.imshow(heatmap_array, aspect='auto', cmap='viridis', vmin=0, vmax=1)

    ax2.set_xlabel("Chunk Index (0-99)", fontsize=12)
    ax2.set_ylabel("Base", fontsize=12)
    ax2.set_title("MSD Filter Effectiveness Heatmap", fontsize=14)
    ax2.set_yticks(range(len(bases_sorted)))
    ax2.set_yticklabels(bases_sorted)

    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label("Mean Filter Effectiveness", fontsize=10)

    plt.tight_layout()

    # Save the heatmap
    heatmap_path = Path("output/msd_filter_effectiveness_heatmap.png")
    plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
    print(f"Heatmap saved to {heatmap_path}")

    # Calculate and print summary table
    print("\n" + "=" * 60)
    print("Summary Table: Mean Filter Effectiveness by Base")
    print("=" * 60)
    print(f"{'Base':<10} {'Mean Effectiveness':<20} {'Sample Count':<15}")
    print("-" * 60)

    summary_data = []
    for base_str, chunks in aggregated_stats.items():
        base = int(base_str)
        total_sum = sum(chunk["sum"] for chunk in chunks)
        total_count = sum(chunk["count"] for chunk in chunks)

        if total_count > 0:
            mean_effectiveness = total_sum / total_count
        else:
            mean_effectiveness = 0.0

        summary_data.append((base, mean_effectiveness, total_count))

    # Sort by base
    summary_data.sort(key=lambda x: x[0])

    for base, mean_eff, count in summary_data:
        print(f"{base:<10} {mean_eff:<20.6f} {count:<15,}")

    print("=" * 60)


if __name__ == "__main__":
    main()
