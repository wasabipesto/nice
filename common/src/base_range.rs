//! A module for calculating the apprpriate range for each base.

use crate::FieldSize;
use malachite::base::num::arithmetic::traits::{CeilingRoot, FloorRoot, Pow};
use malachite::natural::Natural;

/// Get the range of possible values for a base.
/// Returns None if there are no valid numbers in that base.
///
/// **Range semantics**: This represents a half-open range [`range_start`, `range_end`),
/// following Rust's standard convention where `range_start` is inclusive and `range_end` is exclusive.
#[must_use]
pub fn get_base_range_natural(base: u32) -> Option<(Natural, Natural)> {
    let b = Natural::from(base);
    let k = u64::from(base / 5);

    match base % 5 {
        0 => Some((b.clone().pow(3 * k - 1).ceiling_root(3), b.pow(k))),
        1 => None,
        2 => Some((b.clone().pow(k), b.pow(3 * k + 1).floor_root(3))),
        3 => Some((
            b.clone().pow(3 * k + 1).ceiling_root(3),
            b.pow(2 * k + 1).floor_root(2),
        )),
        4 => Some((
            b.clone().pow(2 * k + 1).ceiling_root(2),
            b.pow(3 * k + 2).floor_root(3),
        )),
        _ => None,
    }
}

/// Get the range of possible values for a base, but return u128.
/// Returns None if there are no valid numbers in that base.
/// Returns Err if the numbers are too large for u128.
///
/// **Range semantics**: This represents a half-open range [`range_start`, `range_end`),
/// following Rust's standard convention where `range_start` is inclusive and `range_end` is exclusive.
///
/// # Errors
/// Returns an error if the results are too large for u128.
pub fn get_base_range_u128(base: u32) -> Result<Option<FieldSize>, String> {
    // get the natural results
    let (range_start, range_end) = match get_base_range_natural(base) {
        Some((min, max)) => (
            // convert to u128
            u128::try_from(&min).map_err(|_| format!("Failed to convert {min} to u128."))?,
            u128::try_from(&max).map_err(|_| format!("Failed to convert {max} to u128."))?,
        ),
        None => return Ok(None),
    };
    Ok(Some(FieldSize::new(range_start, range_end)))
}

#[cfg(test)]
#[allow(clippy::too_many_lines)]
mod tests {
    use super::*;
    use std::str::FromStr;

    #[test_log::test]
    fn test_get_base_range_u128() {
        assert_eq!(
            get_base_range_u128(5),
            Ok(Some(FieldSize::new(3u128, 5u128)))
        );
        assert_eq!(get_base_range_u128(6), Ok(None));
        assert_eq!(
            get_base_range_u128(7),
            Ok(Some(FieldSize::new(7u128, 13u128)))
        );
        assert_eq!(
            get_base_range_u128(8),
            Ok(Some(FieldSize::new(16u128, 22u128)))
        );
        assert_eq!(
            get_base_range_u128(9),
            Ok(Some(FieldSize::new(27u128, 38u128)))
        );
        assert_eq!(
            get_base_range_u128(10),
            Ok(Some(FieldSize::new(47u128, 100u128)))
        );
        assert_eq!(
            get_base_range_u128(40),
            Ok(Some(FieldSize::new(
                1_916_284_264_916u128,
                6_553_600_000_000u128
            )))
        );
        assert_eq!(
            get_base_range_u128(80),
            Ok(Some(FieldSize::new(
                653_245_554_420_798_943_087_177_909_799u128,
                2_814_749_767_106_560_000_000_000_000_000u128
            )))
        );
    }

    #[test_log::test]
    fn test_get_base_range_natural() {
        assert_eq!(
            get_base_range_natural(5),
            Some((Natural::from(3u32), Natural::from(5u32)))
        );
        assert_eq!(get_base_range_natural(6), None);
        assert_eq!(
            get_base_range_natural(7),
            Some((Natural::from(7u32), Natural::from(13u32)))
        );
        assert_eq!(
            get_base_range_natural(8),
            Some((Natural::from(16u32), Natural::from(22u32)))
        );
        assert_eq!(
            get_base_range_natural(9),
            Some((Natural::from(27u32), Natural::from(38u32)))
        );
        assert_eq!(
            get_base_range_natural(10),
            Some((Natural::from(47u32), Natural::from(100u32)))
        );
        assert_eq!(
            get_base_range_natural(20),
            Some((Natural::from(58_945u32), Natural::from(160_000u32)))
        );
        assert_eq!(
            get_base_range_natural(30),
            Some((Natural::from(234_613_921u32), Natural::from(729_000_000u32)))
        );
        assert_eq!(
            get_base_range_natural(40),
            Some((
                Natural::from(1_916_284_264_916u64),
                Natural::from(6_553_600_000_000u64)
            ))
        );
        assert_eq!(
            get_base_range_natural(50),
            Some((
                Natural::from(26_507_984_537_059_635u64),
                Natural::from(97_656_250_000_000_000u64)
            ))
        );
        // start getting rounding errors here
        assert_eq!(
            get_base_range_natural(60),
            Some((
                Natural::from(556_029_612_114_824_200_908u128),
                Natural::from(2_176_782_336_000_000_000_000u128)
            ))
        );
        assert_eq!(
            get_base_range_natural(70),
            Some((
                Natural::from(16_456_591_172_673_850_596_148_008u128),
                Natural::from(67_822_307_284_900_000_000_000_000u128)
            ))
        );
        assert_eq!(
            get_base_range_natural(80),
            Some((
                Natural::from(653_245_554_420_798_943_087_177_909_799u128),
                Natural::from(2_814_749_767_106_560_000_000_000_000_000u128)
            ))
        );
        assert_eq!(
            get_base_range_natural(90),
            Some((
                Natural::from(33_492_764_832_792_484_045_981_163_311_105_668u128),
                Natural::from(150_094_635_296_999_121_000_000_000_000_000_000u128)
            ))
        );
        // around here we run into the limits of u128
        assert_eq!(
            get_base_range_natural(100),
            Some((
                Natural::from_str("2154434690031883721759293566519350495260").unwrap(),
                Natural::from_str("10000000000000000000000000000000000000000").unwrap()
            ))
        );
        assert_eq!(
            get_base_range_natural(110),
            Some((
                Natural::from_str("169892749571608053239273597713205371466519752").unwrap(),
                Natural::from_str("814027493868397611133210000000000000000000000").unwrap()
            ))
        );
        assert_eq!(
            get_base_range_natural(120),
            Some((
                Natural::from_str("16117196090075248994613996554363597629408239219454").unwrap(),
                Natural::from_str("79496847203390844133441536000000000000000000000000").unwrap()
            ))
        );
        // run through the mod5 series at a the high end to check everything is still good
        assert_eq!(get_base_range_natural(121), None);
        assert_eq!(
            get_base_range_natural(122),
            Some((
                Natural::from_str("118205024187370033135932935819405317049548439289856").unwrap(),
                Natural::from_str("586258581805989694050980431834549184603056531020210").unwrap()
            ))
        );
        assert_eq!(
            get_base_range_natural(123),
            Some((
                Natural::from_str("715085071699820536699499456671007010425915160419662").unwrap(),
                Natural::from_str("1594686179043939546502781159240976178904795301633107").unwrap()
            ))
        );
        assert_eq!(
            get_base_range_natural(124),
            Some((
                Natural::from_str("1944604500263970232242123784503740458789493393829926").unwrap(),
                Natural::from_str("4342450740818512904293955173690913927483946149220888").unwrap()
            ))
        );
        assert_eq!(
            get_base_range_natural(125),
            Some((
                Natural::from_str("5293955920339377119177015629247762262821197509765625").unwrap(),
                Natural::from_str("26469779601696885595885078146238811314105987548828125").unwrap()
            ))
        );
    }
}
