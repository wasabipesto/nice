//! Expand basic distribution with some redundant stats.

use super::*;

pub fn expand_distribution(
    distributions: &[UniquesDistributionSimple],
    base: u32,
) -> Vec<UniquesDistribution> {
    let total_count = distributions.iter().fold(0, |acc, d| acc + d.count);
    distributions
        .iter()
        .map(|d| UniquesDistribution {
            num_uniques: d.num_uniques,
            count: d.count,
            niceness: d.num_uniques as f32 / base as f32,
            density: d.count as f32 / total_count as f32,
        })
        .collect()
}

pub fn downsample_distributions(
    submissions: &[SubmissionRecord],
    base: u32,
) -> Vec<UniquesDistribution> {
    // set up counter
    let mut counter: HashMap<u32, UniquesDistributionSimple> = HashMap::new();
    for n in 1..=base {
        counter.insert(
            n,
            UniquesDistributionSimple {
                num_uniques: n,
                count: 0,
            },
        );
    }

    // count all submissions
    for sub in submissions {
        if let Some(sub_dist) = &sub.distribution {
            for sub_dist in sub_dist {
                if let Some(counter_dist) = counter.get_mut(&sub_dist.num_uniques) {
                    counter_dist.count += sub_dist.count;
                }
            }
        }
    }

    // collate map values
    let counter_values: Vec<UniquesDistributionSimple> = counter.values().cloned().collect();

    // expand out & return
    expand_distribution(&counter_values, base)
}

pub fn shrink_distribution(distribution: &[UniquesDistribution]) -> Vec<UniquesDistributionSimple> {
    distribution
        .iter()
        .map(|d| UniquesDistributionSimple {
            num_uniques: d.num_uniques,
            count: d.count,
        })
        .collect()
}

// TODO: tests
