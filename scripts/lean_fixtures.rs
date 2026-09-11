#!/usr/bin/env rust-script
//! ```cargo
//! [dependencies]
//! nice_common = { path = "../common" }
//! serde = { version = "1.0", features = ["derive"] }
//! serde_json = "1.0"
//! ```
//! Emit the tables the Lean conformance check compares against the model
//! (`proofs/Conformance.lean`). Small parameters on purpose: the Lean model
//! is executable but not fast, and the point is "model = code" on the exact
//! tables the code builds, not throughput. Regenerate with
//! `just lean-fixtures`; the output is checked in.

use nice_common::base_range::get_base_range_u128;
use nice_common::client_process::{get_is_nice, get_is_nice_with_known_lsd};
use nice_common::lsd_filter::get_valid_multi_lsd_bitmap;
use nice_common::client_process::process_range_niceonly;
use nice_common::gpu_config::{chunk_constants, chunk_constants_u16, prefilter_params};
use nice_common::msd_prefix_filter::{
    get_valid_ranges_recursive, get_valid_ranges_recursive_masked, has_duplicate_msd_prefix,
    MaskedRecursion,
};
use nice_common::FieldSize;
use nice_common::residue_filter::get_residue_filter;
use nice_common::stride_filter::StrideTable;
use serde::Serialize;
use std::fs;
use std::path::PathBuf;

#[derive(Serialize)]
struct Residue {
    base: u32,
    residues: Vec<u32>,
}

#[derive(Serialize)]
struct Lsd {
    base: u32,
    k: u32,
    valid: Vec<u32>,
}

#[derive(Serialize)]
struct FirstValid {
    start: String,
    n: String,
    idx: usize,
}

#[derive(Serialize)]
struct Stride {
    base: u32,
    k: u32,
    modulus: String,
    valid_residues: Vec<u32>,
    gap_table: Vec<u32>,
    /// Digit sets, one sorted list per residue (from `low_digit_masks`).
    low_digits: Vec<Vec<u32>>,
    first_valid: Vec<FirstValid>,
}

#[derive(Serialize)]
struct Range {
    base: u32,
    start: Option<String>,
    end: Option<String>,
}

#[derive(Serialize)]
struct Msd {
    base: u32,
    /// (start, end) half-open, and whether `analyze_range` rejected it.
    verdicts: Vec<(String, String, bool)>,
    /// (start, end, depth, min_size) → emitted leaves as (start, end).
    leaves: Vec<(String, String, u32, String, Vec<(String, String)>)>,
}

#[derive(Serialize)]
struct Pipeline {
    base: u32,
    k: u32,
    start: String,
    end: String,
    /// (depth, min_size) → masked leaves as (start, end, digits).
    masked_leaves: Vec<(u32, String, Vec<(String, String, Vec<u32>)>)>,
    /// `process_range_niceonly` over the whole range (production floor).
    nice: Vec<String>,
}

#[derive(Serialize)]
struct GpuConfig {
    base: u32,
    /// (exponent, base^exponent) below 2^31 and below 2^16.
    chunk: (u32, u32),
    chunk_u16: (u32, u32),
    /// Prefilter digits when enabled, with the base range start.
    prefilter_digits: Option<u32>,
    range_start: Option<String>,
}

#[derive(Serialize)]
struct Seeded {
    base: u32,
    k: u32,
    /// (n, seeded verdict, plain verdict)
    samples: Vec<(String, bool, bool)>,
}

fn digits_of_mask(mask: u64) -> Vec<u32> {
    (0..64).filter(|d| mask & (1u64 << d) != 0).collect()
}

fn main() {
    let out = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("proofs")
        .join("fixtures");
    let out = if out.exists() {
        out
    } else {
        PathBuf::from("proofs/fixtures")
    };
    fs::create_dir_all(&out).unwrap();

    let residues: Vec<Residue> = (5..=40)
        .map(|base| Residue {
            base,
            residues: get_residue_filter(&base),
        })
        .collect();
    fs::write(
        out.join("residue.json"),
        serde_json::to_string_pretty(&residues).unwrap(),
    )
    .unwrap();

    let lsd: Vec<Lsd> = [(10, 1), (10, 2), (12, 2), (16, 2), (10, 3)]
        .iter()
        .map(|&(base, k)| Lsd {
            base,
            k,
            valid: get_valid_multi_lsd_bitmap(base, k)
                .iter()
                .enumerate()
                .filter(|(_, &v)| v)
                .map(|(i, _)| i as u32)
                .collect(),
        })
        .collect();
    fs::write(out.join("lsd.json"), serde_json::to_string_pretty(&lsd).unwrap()).unwrap();

    let stride: Vec<Stride> = [(10, 1), (10, 2), (12, 2), (16, 2)]
        .iter()
        .map(|&(base, k)| {
            let t = StrideTable::new(base, k);
            let starts: Vec<u128> = [0u128, 1, 47, 68, 69, 99, 100, 1_000, 12_345]
                .into_iter()
                .chain((0..8).map(|i| t.modulus * 3 + i * 7 + 1))
                .chain([t.modulus - 1, t.modulus, t.modulus + 1])
                .collect();
            Stride {
                base,
                k,
                modulus: t.modulus.to_string(),
                valid_residues: t.valid_residues.clone(),
                gap_table: t.gap_table.clone(),
                low_digits: t.low_digit_masks.iter().map(|&m| digits_of_mask(m)).collect(),
                first_valid: starts
                    .into_iter()
                    .map(|start| {
                        let (n, idx) = t.first_valid_at_or_after(start);
                        FirstValid {
                            start: start.to_string(),
                            n: n.to_string(),
                            idx,
                        }
                    })
                    .collect(),
            }
        })
        .collect();
    fs::write(
        out.join("stride.json"),
        serde_json::to_string_pretty(&stride).unwrap(),
    )
    .unwrap();

    let ranges: Vec<Range> = (5..=20)
        .map(|base| {
            let r = get_base_range_u128(base).unwrap();
            Range {
                base,
                start: r.map(|f| f.start().to_string()),
                end: r.map(|f| f.end().to_string()),
            }
        })
        .collect();
    fs::write(
        out.join("range.json"),
        serde_json::to_string_pretty(&ranges).unwrap(),
    )
    .unwrap();

    // MSD verdicts and recursion leaves on small bases (the Lean SDR search
    // is brute force, so keep the bases small and the windows short).
    let mut msd = Vec::new();
    for base in [8u32, 10, 12, 14, 17] {
        let range = get_base_range_u128(base).unwrap().unwrap();
        let (rs, re) = (range.start(), range.end());
        let mut verdicts = Vec::new();
        let mut x = rs;
        while x < re {
            for len in [1u128, 2, 3, 5, 8, 13, 21] {
                let end = (x + len).min(re);
                verdicts.push((
                    x.to_string(),
                    end.to_string(),
                    has_duplicate_msd_prefix(FieldSize::new(x, end), base),
                ));
            }
            x += 7;
        }
        let mut leaves = Vec::new();
        for (min_size, depth) in [(2u128, 6u32), (4, 8), (1, 10)] {
            let out: Vec<(String, String)> =
                get_valid_ranges_recursive(FieldSize::new(rs, re), base, 0, depth, min_size, 2)
                    .into_iter()
                    .map(|f| (f.start().to_string(), f.end().to_string()))
                    .collect();
            leaves.push((rs.to_string(), re.to_string(), depth, min_size.to_string(), out));
        }
        msd.push(Msd {
            base,
            verdicts,
            leaves,
        });
    }
    fs::write(out.join("msd.json"), serde_json::to_string_pretty(&msd).unwrap()).unwrap();

    // The masked recursion and the whole pipeline on small bases.
    // Depth k is chosen per base so the Lean side (which recomputes the
    // residue set per candidate) stays under a minute.
    let mut pipeline = Vec::new();
    for (base, k) in [(8u32, 3u32), (10, 3), (12, 2), (14, 1), (17, 1)] {
        let range = get_base_range_u128(base).unwrap().unwrap();
        let (rs, re) = (range.start(), range.end());
        let mut masked_leaves = Vec::new();
        for (min_size, depth) in [(2u128, 6u32), (4, 8), (1, 10)] {
            let mut out = Vec::new();
            get_valid_ranges_recursive_masked(
                FieldSize::new(rs, re),
                &MaskedRecursion {
                    base,
                    fixed_lsd_k: k as usize,
                    max_depth: depth,
                    min_range_size: min_size,
                    subdivision_factor: 2,
                },
                0,
                0,
                &mut out,
            );
            let leaves: Vec<(String, String, Vec<u32>)> = out
                .into_iter()
                .map(|(f, m)| (f.start().to_string(), f.end().to_string(), digits_of_mask(m)))
                .collect();
            masked_leaves.push((depth, min_size.to_string(), leaves));
        }
        let t = StrideTable::new(base, k);
        let nice: Vec<String> = process_range_niceonly(&FieldSize::new(rs, re), base, &t)
            .nice_numbers
            .iter()
            .map(|n| n.number.to_string())
            .collect();
        pipeline.push(Pipeline {
            base,
            k,
            start: rs.to_string(),
            end: re.to_string(),
            masked_leaves,
            nice,
        });
    }
    fs::write(
        out.join("pipeline.json"),
        serde_json::to_string_pretty(&pipeline).unwrap(),
    )
    .unwrap();

    // GPU per-base constants (NUM-6, NUM-7).
    let gpu: Vec<GpuConfig> = (5..=128)
        .map(|base| GpuConfig {
            base,
            chunk: chunk_constants(base),
            chunk_u16: chunk_constants_u16(base),
            prefilter_digits: prefilter_params(base).map(|p| p.digits),
            range_start: get_base_range_u128(base)
                .ok()
                .flatten()
                .map(|f| f.start().to_string()),
        })
        .collect();
    fs::write(
        out.join("gpu_config.json"),
        serde_json::to_string_pretty(&gpu).unwrap(),
    )
    .unwrap();

    // Seeded check on real base-40 candidates: the first 300 stride
    // candidates of the base range plus 69 in base 10 for good measure.
    let mut seeded = Vec::new();
    for &(base, k) in &[(40u32, 3u32), (10, 1)] {
        let t = StrideTable::new(base, k);
        let range = get_base_range_u128(base).unwrap().unwrap();
        let (mut n, mut idx) = t.first_valid_at_or_after(range.start());
        let mut samples = Vec::new();
        while samples.len() < 300 && n < range.end() {
            let mask = t.low_digit_masks[idx];
            samples.push((
                n.to_string(),
                get_is_nice_with_known_lsd(n, base, k, mask),
                get_is_nice(n, base),
            ));
            n += u128::from(t.gap_table[idx]);
            idx = (idx + 1) % t.gap_table.len();
        }
        seeded.push(Seeded { base, k, samples });
    }
    fs::write(
        out.join("seeded.json"),
        serde_json::to_string_pretty(&seeded).unwrap(),
    )
    .unwrap();
    println!("fixtures written to {}", out.display());
}
