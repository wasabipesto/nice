//! Expand basic distribution with some redundant stats.

use super::*;

pub fn expand_distribution(
    distributions: &[UniquesDistributionSimple],
    base: u32,
) -> Vec<UniquesDistribution> {
    let total_count: u128 = distributions.iter().map(|d| d.count).sum();
    assert!(total_count > 0);
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
    // set up counter vec
    // indexed by num_uniques
    let mut counter = vec![
        UniquesDistributionSimple {
            num_uniques: 0,
            count: 0
        };
        base as usize + 1
    ];
    for n in 1..=base {
        counter[n as usize] = UniquesDistributionSimple {
            num_uniques: n,
            count: 0,
        };
    }

    // count all submissions
    for sub in submissions.iter().filter_map(|s| s.distribution.as_deref()) {
        for dist in sub {
            if let Some(counter_dist) = counter.get_mut(dist.num_uniques as usize) {
                counter_dist.count += dist.count;
            }
        }
    }

    // expand out & return
    expand_distribution(&counter[1..], base)
}

pub fn mean_stdev_from_distribution(distribution: &[UniquesDistribution]) -> (f32, f32) {
    let mut mean = 0.0;
    let mut stdev = 0.0;
    let count: u128 = distribution.iter().map(|d| d.count).sum();
    assert!(count > 0);

    for d in distribution {
        mean += d.niceness * d.count as f32;
        stdev += d.count as f32 * d.niceness.powi(2);
    }

    mean /= count as f32;
    stdev = (stdev / count as f32 - mean.powi(2)).sqrt();

    (mean, stdev)
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
