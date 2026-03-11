#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "matplotlib",
#     "numpy",
# ]
# ///
"""
Generate bar charts from filter effectiveness data.
Reads filter_effectiveness.json and creates visualizations of raw percentages
and total eliminated percentages per base.
"""

import json
import matplotlib.pyplot as plt
import numpy as np

def load_data(filename='output/filter_effectiveness.json'):
    """Load the filter effectiveness data from JSON file."""
    with open(filename, 'r') as f:
        return json.load(f)

def plot_filter_effectiveness(data):
    """
    Create a grouped bar chart showing raw percentages and total eliminated percentage.
    """
    bases = [d['base'] for d in data]

    # Extract the raw percentages for each filter
    lsd_raw = [d['lsd_raw_pct'] for d in data]
    residue_raw = [d['residue_raw_pct'] for d in data]
    msd_raw = [d['msd_raw_pct'] for d in data]
    total_elim = [d['total_eliminated_pct'] for d in data]

    # Set up the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    x = np.arange(len(bases))
    width = 0.25

    # First subplot: Individual filter raw percentages
    bars1 = ax1.bar(x - width, lsd_raw, width, label='LSD Raw %', alpha=0.8, color='#2E86AB')
    bars2 = ax1.bar(x, residue_raw, width, label='Residue Raw %', alpha=0.8, color='#A23B72')
    bars3 = ax1.bar(x + width, msd_raw, width, label='MSD Raw %', alpha=0.8, color='#F18F01')

    ax1.set_xlabel('Base', fontsize=12)
    ax1.set_ylabel('Percentage Eliminated (%)', fontsize=12)
    ax1.set_title('Individual Filter Raw Effectiveness by Base', fontsize=14, fontweight='bold', pad=20)
    ax1.set_xticks(x)
    ax1.set_xticklabels(bases)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, 100)

    # Second subplot: Total eliminated percentage
    bars4 = ax2.bar(x, total_elim, width=0.6, label='Total Eliminated %',
                    alpha=0.8, color='#C73E1D')

    ax2.set_xlabel('Base', fontsize=12)
    ax2.set_ylabel('Percentage Eliminated (%)', fontsize=12)
    ax2.set_title('Combined Filter Effectiveness (Total Eliminated) by Base',
                  fontsize=14, fontweight='bold', pad=20)
    ax2.set_xticks(x)
    ax2.set_xticklabels(bases)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, 100)

    # Add value labels on top of bars in second subplot
    for bar in bars4:
        height = bar.get_height()
        ax2.annotate(f'{height:.3f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, -3),
                    textcoords="offset points",
                    ha='center', va='top',
                    fontsize=7, rotation=90)

    plt.tight_layout()
    plt.savefig('output/filter_effectiveness_chart.png', dpi=300, bbox_inches='tight')
    print("Chart saved to output/filter_effectiveness_chart.png")

def main():
    """Main function to load data and generate charts."""
    try:
        data = load_data()
        print(f"Loaded data for {len(data)} bases")
        plot_filter_effectiveness(data)
    except FileNotFoundError:
        print("Error: output/filter_effectiveness.json not found.")
        print("Please run the filter_effectiveness.rs script first to generate the data.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
