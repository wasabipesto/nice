#!/usr/bin/env rust-script
//! ```cargo
//! [dependencies]
//! nice_common = { path = "../common" }
//! malachite = { version = "0.9" }
//! ```
//!
//! Cross-check: the dispatched `has_duplicate_msd_prefix(range, base)` path
//! (now uses fixed-width u128 / U256 internally per #16) must agree with
//! the malachite reference for every random sub-range across all
//! specialized bases. Run after any change to the b40/b50/etc. MSD path.

use malachite::base::num::arithmetic::traits::Pow;
use malachite::base::num::conversion::traits::Digits;
use malachite::natural::Natural;
use nice_common::base_range::get_base_range_u128;
use nice_common::msd_prefix_filter::has_duplicate_msd_prefix;
use nice_common::FieldSize;

fn malachite_ref(range: FieldSize, base: u32) -> bool {
    if range.size() == 1 {
        return false;
    }
    let s_sq = Natural::from(range.first()).pow(2).to_digits_asc(&base);
    let e_sq = Natural::from(range.last()).pow(2).to_digits_asc(&base);
    let s_cu = Natural::from(range.first()).pow(3).to_digits_asc(&base);
    let e_cu = Natural::from(range.last()).pow(3).to_digits_asc(&base);
    // Interval digit-domain (Hall) reference, mirroring msd_prefix_filter:
    // conservative per-position digit domains from the endpoint digit
    // arrays, then reject iff no distinct-digit assignment exists.
    let mut doms: Vec<u64> = Vec::new();
    if s_sq.len() == e_sq.len() {
        collect_domains(base, &s_sq, &e_sq, &mut doms);
    }
    if s_cu.len() == e_cu.len() {
        collect_domains(base, &s_cu, &e_cu, &mut doms);
    }
    if doms.is_empty() {
        return false;
    }
    !has_distinct_assignment(&doms)
}

fn collect_domains(base: u32, xd: &[u32], yd: &[u32], doms: &mut Vec<u64>) {
    let mut diff: i64 = 0;
    for j in (0..xd.len()).rev() {
        diff = diff * i64::from(base) + (i64::from(yd[j]) - i64::from(xd[j]));
        if diff >= i64::from(base) - 1 {
            return;
        }
        let size = (diff as u32) + 1;
        let lo = xd[j];
        let mask: u64 = if lo + size <= base {
            (((1u128 << size) - 1) << lo) as u64
        } else {
            let wrapped = lo + size - base;
            ((((1u128 << (base - lo)) - 1) << lo) | ((1u128 << wrapped) - 1)) as u64
        };
        doms.push(mask);
    }
}

fn has_distinct_assignment(doms: &[u64]) -> bool {
    let m = doms.len();
    if m <= 1 {
        return true;
    }
    let mut union: u64 = 0;
    for &d in doms {
        union |= d;
    }
    if (union.count_ones() as usize) < m {
        return false;
    }
    let mut owner = [usize::MAX; 64];
    for i in 0..m {
        let mut visited: u64 = 0;
        if !augment(i, doms, &mut visited, &mut owner) {
            return false;
        }
    }
    true
}

fn augment(i: usize, doms: &[u64], visited: &mut u64, owner: &mut [usize; 64]) -> bool {
    let mut cand = doms[i] & !*visited;
    while cand != 0 {
        let d = cand.trailing_zeros() as usize;
        *visited |= 1u64 << d;
        if owner[d] == usize::MAX || augment(owner[d], doms, visited, owner) {
            owner[d] = i;
            return true;
        }
        cand = doms[i] & !*visited;
    }
    false
}

fn main() {
    let bases: &[u32] = &[
        40, 42, 43, 44, 45, 47, 48, 49, 50, 52, 53, 54, 55, 57, 58, 59, 60,
    ];
    let mut state: u128 = 0xcafe_babe_dead_beef_1234_5678_9abc_def0;
    let mut rng = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        state
    };

    let mut total_tested = 0;
    let mut total_mismatches = 0;
    for &base in bases {
        let r = match get_base_range_u128(base) {
            Ok(Some(r)) => r,
            _ => continue,
        };
        let n_low = r.start();
        let n_high = r.end();
        let span = n_high - n_low;
        let mut tested = 0;
        let mut mismatches = 0;
        for _ in 0..1000 {
            let s = n_low + (rng() % (span / 2));
            let sz = (rng() % (span / 100).max(1)) + 1;
            let e = (s + sz).min(n_high);
            if e <= s {
                continue;
            }
            let r = FieldSize::new(s, e);
            let r1 = malachite_ref(r, base);
            let r2 = has_duplicate_msd_prefix(r, base);
            if r1 != r2 {
                mismatches += 1;
                if mismatches <= 3 {
                    eprintln!("b{} MISMATCH [{}, {}): ref={} new={}", base, s, e, r1, r2);
                }
            }
            tested += 1;
        }
        println!("b{:2}: {} tested, {} mismatches", base, tested, mismatches);
        total_tested += tested;
        total_mismatches += mismatches;
    }
    println!(
        "---\nTotal: {} tested, {} mismatches",
        total_tested, total_mismatches
    );
    if total_mismatches > 0 {
        std::process::exit(1);
    }
}
