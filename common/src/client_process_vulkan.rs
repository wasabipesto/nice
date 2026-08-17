//! Field processing on the Vulkan backend.
//!
//! Mirrors [`crate::client_process_cuda`] exactly — same batching, same
//! fallback behaviour, same `FieldResults` — over the Vulkan compute backend in
//! [`crate::vulkan`] instead of CUDA. The niceonly host pipeline is not merely
//! mirrored but *shared*: both backends drive [`crate::gpu_niceonly`], which
//! runs the MSD prefix filter and streams range descriptors to whichever device
//! is plugged in behind [`RangeSink`].

#![cfg(feature = "vulkan")]

use crate::client_process::{process_range_detailed, process_range_niceonly};
use crate::gpu_config::gpu_supports_base;
use crate::gpu_niceonly::{GPU_LSD_K, report_field, run_range_pipeline};
use crate::residue_filter;
use crate::stride_filter::StrideTable;
use crate::vulkan::{DetailedRun, NiceonlyRun, VulkanContext};
use crate::{
    CLIENT_VERSION, DataToClient, DataToServer, FieldResults, FieldSize, NiceNumberSimple,
    UniquesDistributionSimple,
};
use anyhow::Result;
use log::{debug, warn};
use std::time::Instant;

/// Candidates per dispatch. Large enough to amortize submission overhead,
/// small enough that a u32 histogram bin cannot overflow before the host
/// drains it — see [`DetailedRun::drain_histogram`].
pub const VULKAN_BATCH_SIZE: u128 = 50_000_000;

/// Near-miss records held on the device per field.
const NEAR_MISS_CAPACITY: usize = 1 << 20;

/// Vulkan implementation of `process_range_detailed`.
///
/// **Range semantics**: half-open [`range_start`, `range_end`).
///
/// # Errors
/// Returns an error on any Vulkan failure or if the near-miss buffer overflows.
pub fn process_range_detailed_vulkan(
    ctx: &VulkanContext,
    range: &FieldSize,
    base: u32,
) -> Result<FieldResults> {
    if !gpu_supports_base(base) {
        warn!("base {base} not supported on GPU, falling back to CPU for this field");
        return Ok(process_range_detailed(range, base));
    }

    let start_time = Instant::now();
    let run = DetailedRun::new(ctx, base, NEAR_MISS_CAPACITY)?;
    let mut histogram = vec![0u128; run.config().hist_bins() as usize];

    for batch in range.chunks(VULKAN_BATCH_SIZE) {
        #[allow(clippy::cast_possible_truncation)]
        run.dispatch(batch.start(), batch.size() as u64)?;
        run.drain_histogram(&mut histogram);
    }

    let mut nice_numbers: Vec<NiceNumberSimple> = run
        .near_misses()?
        .into_iter()
        .map(|(number, num_uniques)| NiceNumberSimple {
            number,
            num_uniques,
        })
        .collect();
    nice_numbers.sort_by_key(|n| n.number);

    let distribution: Vec<UniquesDistributionSimple> = (1..=base)
        .map(|i| UniquesDistributionSimple {
            num_uniques: i,
            count: histogram[i as usize],
        })
        .collect();

    #[allow(clippy::cast_precision_loss)]
    {
        let secs = start_time.elapsed().as_secs_f64();
        debug!(
            "Vulkan detailed b{base}: {:.2e} numbers in {secs:.2}s ({:.2e} n/s), {} near-misses",
            range.size() as f64,
            range.size() as f64 / secs,
            nice_numbers.len(),
        );
    }

    Ok(FieldResults {
        distribution,
        nice_numbers,
    })
}

/// The answer for a residue-empty base, if this is one.
///
/// For `b ≡ 3 mod 4` the residue set `R_b` is empty, which means there are
/// provably no solutions — but it also means the stride table does not return
/// so much as panic when indexed
/// (`stride_filter::first_valid_at_or_after` indexes `valid_residues[idx]`).
/// So this has to be checked before any stride table is built.
///
/// The CUDA path has the same guard inside `process_range_niceonly_cuda`; here
/// it sits ahead of the CPU fallback rather than after it, so it also covers
/// bases the GPU itself cannot take.
fn residue_empty_result(base: u32) -> Option<FieldResults> {
    if residue_filter::get_residue_filter_u128(&base).is_empty() {
        debug!("base {base} is residue-empty; no candidates to check");
        return Some(FieldResults {
            distribution: Vec::new(),
            nice_numbers: Vec::new(),
        });
    }
    None
}

/// Vulkan implementation of `process_range_niceonly`.
///
/// Runs the MSD prefix filter on the CPU (all cores) and checks the surviving
/// ranges' stride-valid candidates on the GPU, which reconstructs them from the
/// residue table on-device. Produces the exact same nice-number set as the CPU
/// path: the coarser MSD floor makes the GPU's candidate set a *superset*, and
/// the per-candidate check is identical.
///
/// **Range semantics**: half-open [`range_start`, `range_end`).
///
/// # Errors
/// Returns an error on any Vulkan failure or if the output buffer overflows.
pub fn process_range_niceonly_vulkan(
    ctx: &VulkanContext,
    range: &FieldSize,
    base: u32,
) -> Result<FieldResults> {
    if let Some(empty) = residue_empty_result(base) {
        return Ok(empty);
    }
    if !gpu_supports_base(base) {
        warn!("base {base} not supported on GPU, falling back to CPU for this field");
        let table = StrideTable::new(base, GPU_LSD_K);
        return Ok(process_range_niceonly(range, base, &table));
    }

    // Build (or fetch cached) the pipeline and device residue table up front,
    // so the pipeline's timings reflect per-field work only.
    let mut run = NiceonlyRun::new(ctx, base, range.start())?;
    let stats = run_range_pipeline(&mut run, range, base)?;
    let nice_numbers = run.finish()?;
    debug!(
        "Vulkan niceonly pipeline: {} ranges in {} dispatches, M={}, R={}, found {}",
        stats.num_ranges,
        stats.launches,
        run.config().stride_m,
        run.config().stride_r,
        nice_numbers.len()
    );
    report_field("Vulkan", base, range, &stats);

    Ok(FieldResults {
        distribution: Vec::new(),
        nice_numbers,
    })
}

/// Process a field on the Vulkan backend (detailed mode).
///
/// # Errors
/// Returns an error on any Vulkan failure.
pub fn process_detailed_vulkan(
    ctx: &VulkanContext,
    claim_data: &DataToClient,
    username: &String,
) -> Result<DataToServer> {
    let results = process_range_detailed_vulkan(ctx, &claim_data.into(), claim_data.base)?;
    Ok(DataToServer {
        claim_id: claim_data.claim_id,
        username: username.to_owned(),
        client_version: CLIENT_VERSION.to_string(),
        unique_distribution: Some(results.distribution),
        nice_numbers: results.nice_numbers,
        telemetry: None,
    })
}

/// Process a field on the Vulkan backend (niceonly mode).
///
/// # Errors
/// Returns an error on any Vulkan failure.
pub fn process_niceonly_vulkan(
    ctx: &VulkanContext,
    claim_data: &DataToClient,
    username: &String,
) -> Result<DataToServer> {
    let results = process_range_niceonly_vulkan(ctx, &claim_data.into(), claim_data.base)?;
    Ok(DataToServer {
        claim_id: claim_data.claim_id,
        username: username.to_owned(),
        client_version: CLIENT_VERSION.to_string(),
        unique_distribution: None,
        nice_numbers: results.nice_numbers,
        telemetry: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The GPU histogram bins are u32; a batch must not be able to overflow one.
    #[test]
    fn batch_size_cannot_overflow_a_u32_bin() {
        assert!(VULKAN_BATCH_SIZE < u128::from(u32::MAX));
    }

    /// Bases with `b % 4 == 3` have an empty residue set, and the stride table
    /// panics rather than returning when indexed. Niceonly must answer "nothing
    /// here" instead of taking the client down, and must decide that *before*
    /// building any stride table.
    #[test]
    fn residue_empty_bases_short_circuit_before_the_stride_table() {
        let mut checked = 0;
        for base in 10..=60 {
            let empty = residue_filter::get_residue_filter_u128(&base).is_empty();
            assert_eq!(
                empty,
                base % 4 == 3,
                "base {base}: residue-emptiness should track b mod 4 == 3"
            );
            match residue_empty_result(base) {
                Some(out) => {
                    assert!(empty, "base {base} short-circuited but has residues");
                    assert!(out.nice_numbers.is_empty() && out.distribution.is_empty());
                    checked += 1;
                }
                None => assert!(!empty, "base {base} is residue-empty but was not caught"),
            }
        }
        assert!(checked >= 10, "only {checked} residue-empty bases exercised");
    }

    /// CPU/GPU parity on the detailed path. Requires a Vulkan device, so it is
    /// `#[ignore]`d — but unlike the CUDA equivalent it also passes against
    /// lavapipe, so it can run without any GPU at all:
    ///
    /// ```text
    /// VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json \
    ///   cargo test -p nice_common --features vulkan -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "requires a Vulkan device"]
    fn vulkan_matches_cpu_detailed() {
        let ctx = VulkanContext::new(0).expect("Vulkan init");
        // 10: single-limb candidates. 40: the production base, two limbs.
        // 62/80: above 64 the digit mask needs a second word, and above 68 the
        // CPU itself leaves its U256 fast path — so these exercise both the
        // codegen branch and the widest agreement we can check.
        for (base, count) in [
            (10u32, 1_000_000u128),
            (40, 2_000_000),
            (62, 200_000),
            (80, 100_000),
        ] {
            let start = crate::base_range::get_base_range_u128(base)
                .unwrap()
                .unwrap()
                .range_start;
            let range = FieldSize::new(start, start + count);
            let gpu = process_range_detailed_vulkan(&ctx, &range, base).expect("vulkan run");
            let cpu = process_range_detailed(&range, base);
            assert_eq!(
                gpu.distribution, cpu.distribution,
                "base {base}: distribution mismatch"
            );
            assert_eq!(
                gpu.nice_numbers, cpu.nice_numbers,
                "base {base}: near-miss mismatch"
            );
            println!("base {base}: {count} candidates match the CPU exactly");
        }
    }

    /// CPU/GPU parity on the niceonly path, over the same range with the same
    /// stride table. The GPU checks a *superset* of the CPU's candidates (its
    /// MSD floor is coarser), so the nice-number sets must still be identical —
    /// that is the whole correctness claim of the pipeline.
    ///
    /// Runs on lavapipe as well as real hardware; see
    /// [`vulkan_matches_cpu_detailed`] for the invocation.
    #[test]
    #[ignore = "requires a Vulkan device"]
    fn vulkan_matches_cpu_niceonly() {
        let ctx = VulkanContext::new(0).expect("Vulkan init");
        // 10 is the base with the known solution; 12 and 25 are small bases
        // where the CUDA path's prefilter is host-disabled (the v3.2.14
        // phantom-zero regression class); 40 is production; 45 and 62 straddle
        // the one-word/two-word digit-mask branch at 64. 80 is production and
        // sits past the k=3 modulus crossing at base 65, so it is the only
        // entry here that exercises the 4-bit Horner chunk on the device —
        // everything below it reduces the offset a byte at a time.
        for base in [10u32, 12, 25, 40, 45, 62, 80] {
            let Ok(Some(base_range)) = crate::base_range::get_base_range_u128(base) else {
                continue;
            };
            let start = base_range.range_start;
            let end = (start + 5_000_000).min(base_range.range_end);
            let range = FieldSize::new(start, end);

            let table = StrideTable::new(base, GPU_LSD_K);
            let mut cpu = process_range_niceonly(&range, base, &table).nice_numbers;
            cpu.sort_by_key(|n| n.number);
            let gpu = process_range_niceonly_vulkan(&ctx, &range, base).expect("vulkan run");

            assert_eq!(cpu, gpu.nice_numbers, "base {base}: niceonly mismatch");
            assert!(
                gpu.distribution.is_empty(),
                "base {base}: niceonly must report no distribution"
            );
            println!(
                "base {base}: [{start}, {end}) agrees, {} nice",
                gpu.nice_numbers.len()
            );
        }
    }

    /// The known solution, through the niceonly path this time: the whole
    /// pipeline (MSD filter, descriptors, on-device residue reconstruction)
    /// has to line up for 69 to come back.
    #[test]
    #[ignore = "requires a Vulkan device"]
    fn vulkan_niceonly_finds_69_in_base_10() {
        let ctx = VulkanContext::new(0).expect("Vulkan init");
        let base_range = crate::base_range::get_base_range_u128(10).unwrap().unwrap();
        let range = FieldSize::new(base_range.range_start, base_range.range_end);
        let results = process_range_niceonly_vulkan(&ctx, &range, 10).expect("vulkan run");
        assert_eq!(
            results
                .nice_numbers
                .iter()
                .map(|n| n.number)
                .collect::<Vec<_>>(),
            vec![69]
        );
    }

    /// The known solution: 69 is nice in base 10.
    #[test]
    #[ignore = "requires a Vulkan device"]
    fn vulkan_finds_69_in_base_10() {
        let ctx = VulkanContext::new(0).expect("Vulkan init");
        let range = FieldSize::new(47, 100);
        let results = process_range_detailed_vulkan(&ctx, &range, 10).expect("vulkan run");
        let hit = results
            .nice_numbers
            .iter()
            .find(|n| n.number == 69)
            .expect("69 not found in base 10");
        assert_eq!(hit.num_uniques, 10);
    }
}
