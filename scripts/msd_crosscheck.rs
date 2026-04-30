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
    if s_sq.len() != e_sq.len() {
        return false;
    }
    let sq_prefix = common_prefix(&s_sq, &e_sq);
    if has_dup(&sq_prefix) {
        return true;
    }
    let s_cu = Natural::from(range.first()).pow(3).to_digits_asc(&base);
    let e_cu = Natural::from(range.last()).pow(3).to_digits_asc(&base);
    if s_cu.len() != e_cu.len() {
        return false;
    }
    let cu_prefix = common_prefix(&s_cu, &e_cu);
    if has_dup(&cu_prefix) {
        return true;
    }
    if has_over(&sq_prefix, &cu_prefix) {
        return true;
    }
    let k = 2usize;
    let b_k = u128::from(base).saturating_pow(k as u32);
    if range.first() / b_k == range.last() / b_k {
        let lsd_sq: Vec<u32> = s_sq.iter().take(k).copied().collect();
        let lsd_cu: Vec<u32> = s_cu.iter().take(k).copied().collect();
        if has_over(&sq_prefix, &lsd_sq)
            || has_over(&cu_prefix, &lsd_cu)
            || has_over(&sq_prefix, &lsd_cu)
            || has_over(&cu_prefix, &lsd_sq)
            || has_dup(&lsd_sq)
            || has_dup(&lsd_cu)
            || has_over(&lsd_sq, &lsd_cu)
        {
            return true;
        }
    }
    false
}

fn common_prefix(d1: &[u32], d2: &[u32]) -> Vec<u32> {
    let l1 = d1.len();
    let l2 = d2.len();
    let mut out = Vec::new();
    for i in 0..l1.min(l2) {
        if d1[l1 - 1 - i] == d2[l2 - 1 - i] {
            out.push(d1[l1 - 1 - i]);
        } else {
            break;
        }
    }
    out
}
fn has_dup(d: &[u32]) -> bool {
    let mut s = vec![false; 256];
    for &x in d {
        if s[x as usize] {
            return true;
        }
        s[x as usize] = true;
    }
    false
}
fn has_over(d1: &[u32], d2: &[u32]) -> bool {
    let mut s = vec![false; 256];
    for &x in d1 {
        s[x as usize] = true;
    }
    for &x in d2 {
        if s[x as usize] {
            return true;
        }
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
