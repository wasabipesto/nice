//! Expand basic distribution with some redundant stats.

use super::*;

pub fn expand_distribution(
    distribution: &[DistributionSimple],
    base: u32,
) -> Vec<UniquesDistribution> {
    let base_f32 = base as f32;
    let total_count_f32 = distribution.iter().fold(0, |acc, d| acc + d.count) as f32;
    distribution
        .iter()
        .map(|d| UniquesDistribution {
            num_uniques: d.num_uniques,
            count: d.count,
            niceness: d.num_uniques as f32 / base_f32,
            density: d.count as f32 / total_count_f32,
        })
        .collect()
}

pub fn shrink_distribution(distribution: &[UniquesDistribution]) -> Vec<DistributionSimple> {
    distribution
        .iter()
        .map(|d| DistributionSimple {
            num_uniques: d.num_uniques,
            count: d.count,
        })
        .collect()
}

// TODO: tests
