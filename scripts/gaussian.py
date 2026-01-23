# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "matplotlib",
#     "numpy",
#     "plotext",
#     "pyqt5",
#     "pyside2",
#     "requests",
#     "scipy",
# ]
# ///

import requests
import json
from scipy import stats
import plotext as plt
import numpy as np


# Get all base data, includes:
# - id (base)
# - range_start
# - range_end
# - range_size
# - checked_detailed
# - checked_niceonly
# - minimum_cl
# - niceness_mean
# - niceness_stdev
# - distribution
#   - count
#   - num_uniques
#   - density (0-1)
#   - niceness (0-1)
# - numbers
#   - number
#   - num_uniques
#   - base
#   - niceness (0-1)
bases = requests.get("https://data.nicenumbers.net/bases", params=[("order","id.asc")]).json()

# Pick the base that has the highest amount searched
base = sorted(bases, key=lambda x: x["checked_detailed"], reverse=True)[0]
base_num = base["id"]
distribution = base["distribution"]

# Plot distribution (niceness vs density)
xvals = [d["niceness"] for d in distribution]
yvals = [d["density"] for d in distribution]
markers = [
    base["niceness_mean"] - 9*base["niceness_stdev"],
    base["niceness_mean"] - 6*base["niceness_stdev"],
    base["niceness_mean"] - 3*base["niceness_stdev"],
    base["niceness_mean"],
    base["niceness_mean"] + 3*base["niceness_stdev"],
    base["niceness_mean"] + 6*base["niceness_stdev"],
]
plt.plot(xvals, yvals)
[plt.vline(m) for m in markers]
plt.theme("clear")
plt.plotsize(120, 20)
plt.xlabel("Niceness")
plt.xticks(markers)
plt.yticks([])
plt.title(f"Base {base['id']} Uniques Distribution")
plt.show()
plt.clear_figure()
print()

print("Gaussian curve fitting:")
# Fit a Gaussian curve using the mean and stdev from the data
niceness_values = np.array([d['niceness'] for d in distribution])
density_values = np.array([d['density'] for d in distribution])

# Calculate expected Gaussian density values
def gaussian(x, mean, std):
    return (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)

expected_density = gaussian(niceness_values, base["niceness_mean"], base["niceness_stdev"])

# Normalize so both curves have same total area for comparison
expected_density_normalized = expected_density * (np.sum(density_values) / np.sum(expected_density))

# Calculate R² (coefficient of determination)
ss_res = np.sum((density_values - expected_density_normalized) ** 2)
ss_tot = np.sum((density_values - np.mean(density_values)) ** 2)
r_squared = 1 - (ss_res / ss_tot)
print(f"  R² (coefficient of determination): {r_squared:.6f}")

# Chi-squared goodness-of-fit test
# Need to ensure expected values are not too small
observed = density_values * np.sum([d['count'] for d in distribution])
expected = expected_density_normalized * np.sum([d['count'] for d in distribution])

# Filter out bins with very low expected counts (< 5) for valid chi-squared test
mask = expected >= 5
observed_filtered = observed[mask]
expected_filtered = expected[mask]

if len(observed_filtered) > 0:
    chi2_stat = np.sum((observed_filtered - expected_filtered) ** 2 / expected_filtered)
    # Degrees of freedom = number of bins - 1 - number of parameters estimated (we used existing mean/std, so -1)
    dof = len(observed_filtered) - 1
    p_value = 1 - stats.chi2.cdf(chi2_stat, dof)
    print(f"  Chi-squared test: χ²={chi2_stat:.4f}, dof={dof}, p-value={p_value:.4f}")
    print(f"    (bins used: {len(observed_filtered)}/{len(observed)}, {len(observed)-len(observed_filtered)} filtered for low expected counts)")
else:
    print(f"  Chi-squared test: Not enough bins with sufficient expected counts")

# Plot actual vs expected Gaussian
plt.plot(niceness_values, expected_density_normalized, label="Expected Gaussian", marker="dot")
plt.plot(niceness_values, density_values, label="Actual Distribution", marker="hd")
plt.theme("clear")
plt.plotsize(120, 20)
plt.xlabel("Niceness")
plt.ylabel("Density")
plt.title(f"Base {base['id']}: Actual vs Gaussian Distribution (R²={r_squared:.6f})")
plt.show()
plt.clear_figure()
print()

print("For a Gaussian distribution:")
# Calculate the z-score for a perfectly nice number
z_score = (1.0 - base["niceness_mean"]) / base["niceness_stdev"]
prob_greater = stats.norm.sf(z_score)
print(f"Probability of a number being 100% nice (Z={z_score:.4f}): {prob_greater:.4e}")
nums_searched = base["checked_niceonly"]
expected_found = nums_searched * prob_greater
nice_nums_found = [d for d in distribution if d["num_uniques"] == base_num][0]["count"]
print(f"  Numbers searched: {nums_searched:.2e}, Expected found: {expected_found:.2f}, Actual nice numbers found: {nice_nums_found}")

# Calculate the z-score for an off-by-one
off_by_one_niceness = (base_num - 1) / base_num
z_score = (off_by_one_niceness - base["niceness_mean"]) / base["niceness_stdev"]
prob_greater = stats.norm.sf(z_score) - prob_greater
print(f"Probability of an off-by-one ({100*off_by_one_niceness:.1f}% nice) (Z={z_score:.4f}): {prob_greater:.4e}")
nums_searched = base["checked_detailed"]
expected_found = nums_searched * prob_greater
off_by_ones_found = [d for d in distribution if d["num_uniques"] == base_num - 1][0]["count"]
print(f"  Numbers searched: {nums_searched:.2e}, Expected found: {expected_found:.2f}, Actual off-by-ones found: {off_by_ones_found}")

# Calculate the z-score for an off-by-two
off_by_two_niceness = (base_num - 2) / base_num
z_score = (off_by_two_niceness - base["niceness_mean"]) / base["niceness_stdev"]
prob_greater = stats.norm.sf(z_score) - prob_greater
print(f"Probability of an off-by-two ({100*off_by_two_niceness:.1f}% nice) (Z={z_score:.4f}): {prob_greater:.4e}")
expected_found = nums_searched * prob_greater
off_by_twos_found = [d for d in distribution if d["num_uniques"] == base_num - 2][0]["count"]
print(f"  Numbers searched: {nums_searched:.2e}, Expected found: {expected_found:.2f}, Actual off-by-twos found: {off_by_twos_found}\n")
