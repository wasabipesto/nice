//! Computation module for recursive MSD filtering with caching.

use anyhow::Result;
use nice_common::FieldSize;
use nice_common::msd_prefix_filter::has_duplicate_msd_prefix;

use crate::db::{DbPool, cache_range, get_cached_range};

/// Recursively compute the valid size of ranges after MSD filtering, with caching.
///
/// This function will:
/// 1. Check the cache for existing results at the required depth
/// 2. If found, return the cached result immediately
/// 3. If not found, compute recursively and cache the result
///
/// # Arguments
/// * `pool` - Database connection pool for caching
/// * `range` - The range to analyze
/// * `base` - The base to check for duplicates
/// * `current_depth` - Current recursion depth
/// * `max_depth` - Maximum recursion depth
/// * `min_range_size` - Minimum range size before stopping recursion
/// * `subdivision_factor` - How many subdivisions to make at each level
pub fn get_valid_ranges_size_recursive(
    pool: &DbPool,
    range: FieldSize,
    base: u32,
    current_depth: u32,
    max_depth: u32,
    min_range_size: u128,
    subdivision_factor: usize,
) -> Result<u128> {
    let range_start = range.start();
    let range_end = range.end();

    // Check cache first - only at depth 0 or when we hit terminal conditions
    // This reduces database overhead significantly
    let is_terminal = current_depth >= max_depth
        || range.size() <= min_range_size
        || range.size() < min_range_size * (subdivision_factor as u128);

    let should_check_cache = current_depth == 0 || is_terminal;

    if should_check_cache {
        if let Some(cached) = get_cached_range(pool, base, range_start, range_end, max_depth)? {
            return Ok(cached.valid_size);
        }
    }

    // Check if range is too small or we've hit max depth
    if current_depth >= max_depth {
        let size = range.size();
        // Only cache if at a significant depth interval to reduce writes
        if current_depth % 5 == 0 || current_depth == max_depth {
            cache_range(
                pool,
                base,
                range_start,
                range_end,
                max_depth,
                min_range_size,
                subdivision_factor,
                size,
            )?;
        }
        return Ok(size);
    }

    if range.size() <= min_range_size {
        let size = range.size();
        // Cache terminal nodes
        cache_range(
            pool,
            base,
            range_start,
            range_end,
            max_depth,
            min_range_size,
            subdivision_factor,
            size,
        )?;
        return Ok(size);
    }

    // Check if the entire range can be skipped
    if has_duplicate_msd_prefix(range, base) {
        // Always cache filtered ranges - important for effectiveness
        cache_range(
            pool,
            base,
            range_start,
            range_end,
            max_depth,
            min_range_size,
            subdivision_factor,
            0,
        )?;
        return Ok(0);
    }

    // Check if subdivision would be worthwhile
    if range.size() < min_range_size * (subdivision_factor as u128) {
        let size = range.size();
        cache_range(
            pool,
            base,
            range_start,
            range_end,
            max_depth,
            min_range_size,
            subdivision_factor,
            size,
        )?;
        return Ok(size);
    }

    // Subdivide the range and recursively check each part
    let chunk_size = range.size() / (subdivision_factor as u128);
    let mut total_size = 0u128;

    for i in 0..subdivision_factor {
        let sub_start = range_start + (i as u128) * chunk_size;
        let sub_end = if i == subdivision_factor - 1 {
            range_end // Last chunk gets any remainder
        } else {
            sub_start + chunk_size
        };

        if sub_start < sub_end {
            let sub_range = FieldSize::new(sub_start, sub_end);
            let sub_size = get_valid_ranges_size_recursive(
                pool,
                sub_range,
                base,
                current_depth + 1,
                max_depth,
                min_range_size,
                subdivision_factor,
            )?;
            total_size += sub_size;
        }
    }

    // Only cache at depth 0 (top level) and at regular intervals to reduce overhead
    if current_depth == 0 || current_depth % 5 == 0 {
        cache_range(
            pool,
            base,
            range_start,
            range_end,
            max_depth,
            min_range_size,
            subdivision_factor,
            total_size,
        )?;
    }

    Ok(total_size)
}

/// Compute the valid size for a base range with caching.
/// This is the main entry point for computing a single base.
pub fn compute_base(
    pool: &DbPool,
    base: u32,
    max_depth: u32,
    min_range_size: u128,
    subdivision_factor: usize,
) -> Result<u128> {
    let base_range = match nice_common::base_range::get_base_range_u128(base)? {
        Some(range) => range,
        None => return Ok(0), // No valid range for this base
    };

    get_valid_ranges_size_recursive(
        pool,
        base_range,
        base,
        0,
        max_depth,
        min_range_size,
        subdivision_factor,
    )
}
