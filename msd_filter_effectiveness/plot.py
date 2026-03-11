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
from tqdm import tqdm
from pathlib import Path


def main():
    # Get base range start/end from output/base_data.json
    base_data_path = Path("output/base_data.json")
    with open(base_data_path) as f:
        base_data = json.load(f)

    # Create a dictionary for quick lookup
    base_ranges = {item["base"]: (item["start"], item["end"]) for item in base_data}
    print("Base data loaded.")

    # Build base range chunks by dividing base range into 100 chunks
    num_chunks = 100
    base_chunks = {}

    for base, (start, end) in base_ranges.items():
        chunk_size = (end - start) / num_chunks
        chunks = []
        for i in range(num_chunks):
            chunk_start = start + i * chunk_size
            chunk_end = start + (i + 1) * chunk_size
            chunks.append((chunk_start, chunk_end))
        base_chunks[base] = chunks
    print("Chunks generated.")

    # Load JSONL data file and aggregate statistics per chunk
    jsonl_path = Path("output/msd_filter_samples.jsonl")

    # Initialize aggregation structure: {base: [(sum, count), ...]}
    chunk_stats = {}

    # Count total lines for progress bar
    print("Counting lines...")
    with open(jsonl_path) as f:
        total_lines = sum(1 for _ in f)
    print(f"Processing {total_lines:,} lines...")

    with open(jsonl_path) as f:
        for line in tqdm(f, total=total_lines):
            if line.strip():
                try:
                    sample = json.loads(line)
                    base = sample["base"]

                    if base not in base_chunks:
                        continue

                    # Initialize chunk stats for this base if needed
                    if base not in chunk_stats:
                        chunk_stats[base] = [[0.0, 0] for _ in range(num_chunks)]

                    num_start = sample["num_start"]
                    effectiveness = sample["effectiveness"]

                    # Find which chunk this sample belongs to
                    chunks = base_chunks[base]
                    for i, (chunk_start, chunk_end) in enumerate(chunks):
                        if chunk_start <= num_start < chunk_end:
                            chunk_stats[base][i][0] += effectiveness
                            chunk_stats[base][i][1] += 1
                            break
                except json.JSONDecodeError:
                    # Skip malformed lines (e.g., incomplete writes)
                    continue
    print("Samples loaded and aggregated.")

    # Calculate mean effectiveness for each chunk
    chunk_effectiveness = {}
    for base, stats in chunk_stats.items():
        means = []
        for sum_val, count in stats:
            if count > 0:
                means.append(sum_val / count)
            else:
                means.append(np.nan)
        chunk_effectiveness[base] = means

    # Plot filter effectiveness per chunk for all bases
    fig, ax = plt.subplots(figsize=(14, 8))

    # Get sorted bases
    bases = sorted(chunk_effectiveness.keys())

    # Create a color map
    colors = plt.cm.viridis(np.linspace(0, 1, len(bases)))

    for idx, base in enumerate(bases):
        means = chunk_effectiveness[base]
        x = np.arange(num_chunks)

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
    ax.set_xlim(0, num_chunks - 1)
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


if __name__ == "__main__":
    main()
