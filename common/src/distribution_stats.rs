//! Expand basic distribution with some redundant stats.

use crate::{SubmissionRecord, UniquesDistribution, UniquesDistributionSimple};

/// Converts a list of `UniquesDistributionSimple` to `UniquesDistribution` by adding
/// some redundant information that's helpful for other tools.
///
/// # Panics
/// Panics if the distribution is empty.
#[must_use]
#[allow(clippy::cast_precision_loss)]
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

/// Incrementally aggregates distribution counts over batches of submissions.
///
/// Folding batch-by-batch produces the same result as one pass over the
/// concatenation, which is what lets the jobs binary stream a base's canon
/// submissions chunk-by-chunk instead of holding them all in memory.
pub struct DistributionAccumulator {
    base: u32,
    // Counter vec indexed by `num_uniques`.
    // Note: Array size is (base + 1) to allow indexing from 0..=base
    // We use indices [1..=base] (inclusive range) since `num_uniques` ranges from 1 to base
    counter: Vec<UniquesDistributionSimple>,
}

impl DistributionAccumulator {
    #[must_use]
    pub fn new(base: u32) -> Self {
        let mut counter = vec![
            UniquesDistributionSimple {
                num_uniques: 0,
                count: 0
            };
            base as usize + 1
        ];
        // Initialize entries for `num_uniques` in [1, base] (inclusive on both ends)
        for n in 1..=base {
            counter[n as usize].num_uniques = n;
        }
        Self { base, counter }
    }

    /// Add a batch of submissions' distribution counts.
    pub fn fold(&mut self, submissions: &[SubmissionRecord]) {
        for sub in submissions.iter().filter_map(|s| s.distribution.as_deref()) {
            for dist in sub {
                if let Some(counter_dist) = self.counter.get_mut(dist.num_uniques as usize) {
                    counter_dist.count += dist.count;
                }
            }
        }
    }

    /// Total count folded so far. Zero means no data: `finalize` would panic
    /// (via `expand_distribution`'s empty assertion), so check this first.
    #[must_use]
    pub fn total_count(&self) -> u128 {
        self.counter.iter().map(|c| c.count).sum()
    }

    /// Expand the accumulated counts into a full distribution.
    ///
    /// # Panics
    /// Panics if nothing was folded in (see `total_count`).
    #[must_use]
    pub fn finalize(&self) -> Vec<UniquesDistribution> {
        // Note: counter[1..] is a half-open range slice that includes indices [1, base],
        // effectively skipping counter[0] which was just a placeholder
        expand_distribution(&self.counter[1..], self.base)
    }
}

/// Take a bunch of `SubmissionRecords`, which each have their own `UniquesDistributions`,
/// and aggregate the total count per `um_uniques`.
#[must_use]
pub fn downsample_distributions(
    submissions: &[SubmissionRecord],
    base: u32,
) -> Vec<UniquesDistribution> {
    let mut acc = DistributionAccumulator::new(base);
    acc.fold(submissions);
    acc.finalize()
}

/// Convert a set of `UniquesDistributions` to a mean and standard deviation.
///
/// # Panics
/// Panics if the distribution is empty.
#[must_use]
#[allow(clippy::cast_precision_loss)]
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

/// Removes some information from a list of `UniquesDistribution` to make `UniquesDistributionSimple`.
#[must_use]
pub fn shrink_distribution(distribution: &[UniquesDistribution]) -> Vec<UniquesDistributionSimple> {
    distribution
        .iter()
        .map(|d| UniquesDistributionSimple {
            num_uniques: d.num_uniques,
            count: d.count,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SearchMode;
    use chrono::Utc;

    /// Build a submission whose distribution has the given counts for
    /// `num_uniques` `1..=counts.len()`.
    fn sub_with_counts(counts: &[u128]) -> SubmissionRecord {
        let base = u32::try_from(counts.len()).unwrap();
        let dist: Vec<UniquesDistributionSimple> = counts
            .iter()
            .enumerate()
            .map(|(i, &count)| UniquesDistributionSimple {
                num_uniques: u32::try_from(i).unwrap() + 1,
                count,
            })
            .collect();
        SubmissionRecord {
            submission_id: 1,
            claim_id: 1,
            field_id: 1,
            search_mode: SearchMode::Detailed,
            submit_time: Utc::now(),
            elapsed_secs: 1.0,
            username: "test".to_string(),
            user_ip: "test".to_string(),
            client_version: "test".to_string(),
            disqualified: false,
            distribution: Some(expand_distribution(&dist, base)),
            numbers: Vec::new(),
        }
    }

    /// Folding batches through the accumulator must equal one pass over the
    /// concatenation - the property that lets the jobs binary stream a base's
    /// submissions chunk-by-chunk.
    #[test]
    fn accumulator_matches_single_pass() {
        let base = 4;
        let subs: Vec<SubmissionRecord> = vec![
            sub_with_counts(&[5, 10, 100, 3]),
            sub_with_counts(&[1, 2, 3, 4]),
            sub_with_counts(&[0, 7, 70, 1]),
            sub_with_counts(&[9, 0, 50, 2]),
            sub_with_counts(&[4, 4, 4, 4]),
        ];

        let single = downsample_distributions(&subs, base);

        let mut acc = DistributionAccumulator::new(base);
        acc.fold(&subs[..2]);
        acc.fold(&subs[2..3]);
        acc.fold(&subs[3..]);
        let folded = acc.finalize();

        assert_eq!(single, folded);
        assert_eq!(
            acc.total_count(),
            subs.iter()
                .flat_map(|s| s.distribution.as_deref().unwrap())
                .map(|d| d.count)
                .sum::<u128>()
        );
    }

    fn create_test_distribution_simple() -> Vec<UniquesDistributionSimple> {
        vec![
            UniquesDistributionSimple {
                num_uniques: 1,
                count: 100,
            },
            UniquesDistributionSimple {
                num_uniques: 2,
                count: 50,
            },
            UniquesDistributionSimple {
                num_uniques: 3,
                count: 25,
            },
        ]
    }

    fn create_test_submissions() -> Vec<SubmissionRecord> {
        let distribution1 = vec![
            UniquesDistribution {
                num_uniques: 1,
                count: 100,
                niceness: 0.1,
                density: 0.5,
            },
            UniquesDistribution {
                num_uniques: 2,
                count: 100,
                niceness: 0.2,
                density: 0.5,
            },
        ];

        let distribution2 = vec![
            UniquesDistribution {
                num_uniques: 1,
                count: 50,
                niceness: 0.1,
                density: 0.25,
            },
            UniquesDistribution {
                num_uniques: 3,
                count: 150,
                niceness: 0.3,
                density: 0.75,
            },
        ];

        vec![
            SubmissionRecord {
                submission_id: 1,
                claim_id: 1,
                field_id: 1,
                search_mode: SearchMode::Detailed,
                submit_time: Utc::now(),
                elapsed_secs: 10.0,
                username: "test1".to_string(),
                user_ip: "127.0.0.1".to_string(),
                client_version: "1.0.0".to_string(),
                disqualified: false,
                distribution: Some(distribution1),
                numbers: vec![],
            },
            SubmissionRecord {
                submission_id: 2,
                claim_id: 2,
                field_id: 1,
                search_mode: SearchMode::Detailed,
                submit_time: Utc::now(),
                elapsed_secs: 15.0,
                username: "test2".to_string(),
                user_ip: "127.0.0.1".to_string(),
                client_version: "1.0.0".to_string(),
                disqualified: false,
                distribution: Some(distribution2),
                numbers: vec![],
            },
        ]
    }

    #[test_log::test]
    #[allow(clippy::float_cmp)]
    fn test_expand_distribution() {
        let simple_dist = create_test_distribution_simple();
        let base = 10;
        let expanded = expand_distribution(&simple_dist, base);

        assert_eq!(expanded.len(), 3);

        // Check first entry
        assert_eq!(expanded[0].num_uniques, 1);
        assert_eq!(expanded[0].count, 100);
        assert_eq!(expanded[0].niceness, 0.1); // 1/10
        assert_eq!(expanded[0].density, 100.0 / 175.0); // 100/(100+50+25)

        // Check second entry
        assert_eq!(expanded[1].num_uniques, 2);
        assert_eq!(expanded[1].count, 50);
        assert_eq!(expanded[1].niceness, 0.2); // 2/10
        assert_eq!(expanded[1].density, 50.0 / 175.0);

        // Check third entry
        assert_eq!(expanded[2].num_uniques, 3);
        assert_eq!(expanded[2].count, 25);
        assert_eq!(expanded[2].niceness, 0.3); // 3/10
        assert_eq!(expanded[2].density, 25.0 / 175.0);
    }

    #[test_log::test]
    #[should_panic(expected = "assertion failed")]
    fn test_expand_distribution_empty() {
        let empty_dist = vec![];
        let _ = expand_distribution(&empty_dist, 10);
    }

    #[test_log::test]
    fn test_downsample_distributions() {
        let submissions = create_test_submissions();
        let base = 10;
        let result = downsample_distributions(&submissions, base);

        // Should have entries for indices 1-10 (base = 10)
        assert_eq!(result.len(), base as usize);

        // Check that counts were aggregated correctly
        // From submission 1: num_uniques=1 has count=100, num_uniques=2 has count=100
        // From submission 2: num_uniques=1 has count=50, num_uniques=3 has count=150
        // So: num_uniques=1 should have count=150, num_uniques=2 should have count=100, num_uniques=3 should have count=150

        assert_eq!(result[0].num_uniques, 1); // index 1-1 = 0
        assert_eq!(result[0].count, 150);

        assert_eq!(result[1].num_uniques, 2); // index 2-1 = 1
        assert_eq!(result[1].count, 100);

        assert_eq!(result[2].num_uniques, 3); // index 3-1 = 2
        assert_eq!(result[2].count, 150);

        // Other entries should have count 0
        for dist in result.iter().skip(3) {
            assert_eq!(dist.count, 0);
        }
    }

    #[test_log::test]
    #[should_panic(expected = "assertion failed: total_count > 0")]
    fn test_downsample_distributions_empty_submissions() {
        let submissions = vec![];
        let base = 10;
        let _result = downsample_distributions(&submissions, base);
    }

    #[test_log::test]
    fn test_mean_stdev_from_distribution() {
        let distribution = vec![
            UniquesDistribution {
                num_uniques: 1,
                count: 100,
                niceness: 0.1,
                density: 0.5,
            },
            UniquesDistribution {
                num_uniques: 2,
                count: 100,
                niceness: 0.2,
                density: 0.5,
            },
        ];

        let (mean, stdev) = mean_stdev_from_distribution(&distribution);

        // Expected mean: (0.1 * 100 + 0.2 * 100) / 200 = 0.15
        assert!((mean - 0.15).abs() < 1e-6);

        // Expected variance: ((0.1^2 * 100 + 0.2^2 * 100) / 200) - 0.15^2
        // = (0.01 * 100 + 0.04 * 100) / 200 - 0.0225 = 0.025 - 0.0225 = 0.0025
        // stdev = sqrt(0.0025) = 0.05
        assert!((stdev - 0.05).abs() < 1e-6);
    }

    #[test_log::test]
    #[should_panic(expected = "assertion failed")]
    fn test_mean_stdev_from_distribution_empty() {
        let empty_dist = vec![];
        let _ = mean_stdev_from_distribution(&empty_dist);
    }

    #[test_log::test]
    fn test_shrink_distribution() {
        let distribution = vec![
            UniquesDistribution {
                num_uniques: 1,
                count: 100,
                niceness: 0.1,
                density: 0.5,
            },
            UniquesDistribution {
                num_uniques: 2,
                count: 50,
                niceness: 0.2,
                density: 0.25,
            },
        ];

        let shrunk = shrink_distribution(&distribution);

        assert_eq!(shrunk.len(), 2);
        assert_eq!(shrunk[0].num_uniques, 1);
        assert_eq!(shrunk[0].count, 100);
        assert_eq!(shrunk[1].num_uniques, 2);
        assert_eq!(shrunk[1].count, 50);
    }

    #[test_log::test]
    fn test_expand_shrink_roundtrip() {
        let original = create_test_distribution_simple();
        let base = 10;
        let expanded = expand_distribution(&original, base);
        let shrunk = shrink_distribution(&expanded);

        assert_eq!(original, shrunk);
    }
}
