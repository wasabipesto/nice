//! A module for dealing with residue filters
//! For more information: <https://beautifulthorns.wixsite.com/home/post/progress-update-on-the-search-for-nice-numbers>

/// Get a list of residue filters for a base.
#[must_use]
pub fn get_residue_filter(base: &u32) -> Vec<u32> {
    let target_residue = base * (base - 1) / 2 % (base - 1);
    (0..(base - 1))
        .filter(|num| (num.pow(2) + num.pow(3)) % (base - 1) == target_residue)
        .collect()
}

/// Get a list of residue filters for a base, but as u128 for easier processing.
#[must_use]
pub fn get_residue_filter_u128(base: &u32) -> Vec<u128> {
    get_residue_filter(base)
        .iter()
        .map(|num| u128::from(*num))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_log::test]
    fn test_get_residue_filter() {
        assert_eq!(get_residue_filter(&10), Vec::from([0, 3, 6, 8]));
        assert_eq!(get_residue_filter(&11), Vec::<u32>::new());
        assert_eq!(get_residue_filter(&12), Vec::from([0, 10]));
        assert_eq!(get_residue_filter(&13), Vec::from([5, 9]));
        assert_eq!(get_residue_filter(&14), Vec::from([0, 12]));
        assert_eq!(get_residue_filter(&15), Vec::<u32>::new());
        assert_eq!(get_residue_filter(&16), Vec::from([0, 5, 9, 14]));
        assert_eq!(get_residue_filter(&17), Vec::from([7]));
        assert_eq!(get_residue_filter(&18), Vec::from([0, 16]));
        assert_eq!(get_residue_filter(&19), Vec::<u32>::new());
        assert_eq!(get_residue_filter(&20), Vec::from([0, 18]));
        assert_eq!(get_residue_filter(&21), Vec::from([5, 9]));
        assert_eq!(get_residue_filter(&22), Vec::from([0, 6, 14, 20]));
        assert_eq!(get_residue_filter(&23), Vec::<u32>::new());
        assert_eq!(get_residue_filter(&24), Vec::from([0, 22]));
        assert_eq!(get_residue_filter(&25), Vec::from([2, 3, 6, 11, 14, 18]));
        assert_eq!(get_residue_filter(&26), Vec::from([0, 5, 10, 15, 20, 24]));
        assert_eq!(get_residue_filter(&27), Vec::<u32>::new());
        assert_eq!(get_residue_filter(&28), Vec::from([0, 9, 18, 26]));
        assert_eq!(get_residue_filter(&29), Vec::from([13, 21]));
        assert_eq!(get_residue_filter(&30), Vec::from([0, 28]));
        assert_eq!(get_residue_filter(&40), Vec::from([0, 12, 26, 38]));
        assert_eq!(
            get_residue_filter(&50),
            Vec::from([0, 7, 14, 21, 28, 35, 42, 48])
        );
        assert_eq!(get_residue_filter(&60), Vec::from([0, 58]));
        assert_eq!(get_residue_filter(&70), Vec::from([0, 23, 45, 68]));
        assert_eq!(get_residue_filter(&80), Vec::from([0, 78]));
        assert_eq!(get_residue_filter(&90), Vec::from([0, 88]));
        assert_eq!(
            get_residue_filter(&100),
            Vec::from([0, 21, 33, 44, 54, 66, 87, 98])
        );
        assert_eq!(get_residue_filter(&110), Vec::from([0, 108]));
        assert_eq!(get_residue_filter(&111), Vec::<u32>::new());
        assert_eq!(get_residue_filter(&112), Vec::from([0, 36, 74, 110]));
        assert_eq!(get_residue_filter(&113), Vec::from([7, 55]));
        assert_eq!(get_residue_filter(&114), Vec::from([0, 112]));
        assert_eq!(get_residue_filter(&115), Vec::<u32>::new());
        assert_eq!(get_residue_filter(&116), Vec::from([0, 45, 69, 114]));
        assert_eq!(get_residue_filter(&117), Vec::from([29, 57]));
        assert_eq!(
            get_residue_filter(&118),
            Vec::from([0, 12, 26, 39, 51, 78, 90, 116])
        );
        assert_eq!(get_residue_filter(&119), Vec::<u32>::new());
        assert_eq!(get_residue_filter(&120), Vec::from([0, 34, 84, 118]));
    }

    /// Closed-form count of valid residues, from the 2026-08 theory review
    /// (scratchpad/2026-08-theory-review/THEORY_AND_SEARCH_DIRECTIONS.md §2).
    ///
    /// The filter keeps n mod (b-1) with n²(n+1) ≡ b(b-1)/2 (mod b-1). By CRT
    /// over the prime powers p^a || b-1:
    /// - even b: each prime power contributes p^⌊a/2⌋ + 1 residues
    /// - b ≡ 3 (mod 4): no solutions (the target m/2 is odd but n²(n+1) is
    ///   always even), so every such base is dead
    /// - b ≡ 1 (mod 4): odd prime powers as above; the 2^a component
    ///   contributes 1 if a is even, 2^((a-1)/2) + 1 if a is odd
    fn predicted_residue_count(b: u64) -> u64 {
        let mut m = b - 1;
        let mut factors = Vec::new();
        let mut p = 2u64;
        while p * p <= m {
            if m.is_multiple_of(p) {
                let mut a = 0u32;
                while m.is_multiple_of(p) {
                    m /= p;
                    a += 1;
                }
                factors.push((p, a));
            }
            p += 1;
        }
        if m > 1 {
            factors.push((m, 1));
        }
        if b.is_multiple_of(2) {
            factors.iter().map(|&(p, a)| p.pow(a / 2) + 1).product()
        } else if b % 4 == 3 {
            0
        } else {
            factors
                .iter()
                .map(|&(p, a)| {
                    if p == 2 {
                        if a % 2 == 0 { 1 } else { 2u64.pow((a - 1) / 2) + 1 }
                    } else {
                        p.pow(a / 2) + 1
                    }
                })
                .product()
        }
    }

    #[test_log::test]
    fn test_residue_count_matches_closed_form() {
        // Independent oracle: the enumerated residue count must match the
        // closed-form classification for every base. This also pins down the
        // theorem that every base ≡ 3 (mod 4) has no valid residues.
        for b in 5u32..=512 {
            assert_eq!(
                get_residue_filter(&b).len() as u64,
                predicted_residue_count(u64::from(b)),
                "residue count mismatch at base {b}"
            );
        }
    }

    #[test_log::test]
    fn test_get_residue_filter_u128() {
        assert_eq!(get_residue_filter_u128(&10), Vec::from([0, 3, 6, 8]));
        assert_eq!(get_residue_filter_u128(&11), Vec::<u128>::new());
        assert_eq!(get_residue_filter_u128(&12), Vec::from([0, 10]));
        assert_eq!(get_residue_filter_u128(&13), Vec::from([5, 9]));
        assert_eq!(get_residue_filter_u128(&14), Vec::from([0, 12]));
        assert_eq!(get_residue_filter_u128(&15), Vec::<u128>::new());
        assert_eq!(get_residue_filter_u128(&16), Vec::from([0, 5, 9, 14]));
        assert_eq!(get_residue_filter_u128(&17), Vec::from([7]));
        assert_eq!(get_residue_filter_u128(&18), Vec::from([0, 16]));
        assert_eq!(get_residue_filter_u128(&19), Vec::<u128>::new());
        assert_eq!(get_residue_filter_u128(&20), Vec::from([0, 18]));
        assert_eq!(get_residue_filter_u128(&21), Vec::from([5, 9]));
        assert_eq!(get_residue_filter_u128(&22), Vec::from([0, 6, 14, 20]));
        assert_eq!(get_residue_filter_u128(&23), Vec::<u128>::new());
        assert_eq!(get_residue_filter_u128(&24), Vec::from([0, 22]));
        assert_eq!(
            get_residue_filter_u128(&25),
            Vec::from([2, 3, 6, 11, 14, 18])
        );
        assert_eq!(
            get_residue_filter_u128(&26),
            Vec::from([0, 5, 10, 15, 20, 24])
        );
        assert_eq!(get_residue_filter_u128(&27), Vec::<u128>::new());
        assert_eq!(get_residue_filter_u128(&28), Vec::from([0, 9, 18, 26]));
        assert_eq!(get_residue_filter_u128(&29), Vec::from([13, 21]));
        assert_eq!(get_residue_filter_u128(&30), Vec::from([0, 28]));
        assert_eq!(get_residue_filter_u128(&40), Vec::from([0, 12, 26, 38]));
        assert_eq!(
            get_residue_filter_u128(&50),
            Vec::from([0, 7, 14, 21, 28, 35, 42, 48])
        );
        assert_eq!(get_residue_filter_u128(&60), Vec::from([0, 58]));
        assert_eq!(get_residue_filter_u128(&70), Vec::from([0, 23, 45, 68]));
        assert_eq!(get_residue_filter_u128(&80), Vec::from([0, 78]));
        assert_eq!(get_residue_filter_u128(&90), Vec::from([0, 88]));
        assert_eq!(
            get_residue_filter_u128(&100),
            Vec::from([0, 21, 33, 44, 54, 66, 87, 98])
        );
        assert_eq!(get_residue_filter_u128(&110), Vec::from([0, 108]));
        assert_eq!(get_residue_filter_u128(&111), Vec::<u128>::new());
        assert_eq!(get_residue_filter_u128(&112), Vec::from([0, 36, 74, 110]));
        assert_eq!(get_residue_filter_u128(&113), Vec::from([7, 55]));
        assert_eq!(get_residue_filter_u128(&114), Vec::from([0, 112]));
        assert_eq!(get_residue_filter_u128(&115), Vec::<u128>::new());
        assert_eq!(get_residue_filter_u128(&116), Vec::from([0, 45, 69, 114]));
        assert_eq!(get_residue_filter_u128(&117), Vec::from([29, 57]));
        assert_eq!(
            get_residue_filter_u128(&118),
            Vec::from([0, 12, 26, 39, 51, 78, 90, 116])
        );
        assert_eq!(get_residue_filter_u128(&119), Vec::<u128>::new());
        assert_eq!(get_residue_filter_u128(&120), Vec::from([0, 34, 84, 118]));
    }
}
