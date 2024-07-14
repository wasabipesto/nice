//! Safe-ish conversions between rust and sql types.
//! Ideally this will be ripped out and implemented as whatever custom diesel types end up being necessary.

use super::*;

pub fn i64_to_u128(i: i64) -> Result<u128, String> {
    if i < 0 {
        Err("i64 value is negative and cannot be converted to u128".to_string())
    } else {
        Ok(i as u128)
    }
}
pub fn u128_to_i64(i: u128) -> Result<i64, String> {
    if i > i64::MAX as u128 {
        Err("u128 value exceeds i64::MAX and cannot be converted to i64".to_string())
    } else {
        Ok(i as i64)
    }
}

/*
pub fn i64_to_u32(i: i64) -> Result<u32, String> {
    if i < 0 {
        Err("i64 value is negative and cannot be converted to u32".to_string())
    } else {
        Ok(i as u32)
    }
}
pub fn u32_to_i64(i: u32) -> Result<i64, String> {
    Ok(i as i64)
}
*/

pub fn i32_to_u32(i: i32) -> Result<u32, String> {
    if i < 0 {
        Err("i32 value is negative and cannot be converted to u32".to_string())
    } else {
        Ok(i as u32)
    }
}
pub fn u32_to_i32(i: u32) -> Result<i32, String> {
    if i > i32::MAX as u32 {
        Err("u32 value exceeds i32::MAX and cannot be converted to i32".to_string())
    } else {
        Ok(i as i32)
    }
}

pub fn i32_to_u8(i: i32) -> Result<u8, String> {
    if i < 0 || i > u8::MAX as i32 {
        Err("i32 value is out of range for u8".to_string())
    } else {
        Ok(i as u8)
    }
}
pub fn u8_to_i32(i: u8) -> Result<i32, String> {
    Ok(i as i32)
}

pub fn bigdec_to_u128(i: BigDecimal) -> Result<u128, String> {
    match i.to_u128() {
        Some(value) => Ok(value),
        None => Err("BigDecimal value cannot be converted to u128".to_string()),
    }
}
pub fn u128_to_bigdec(i: u128) -> Result<BigDecimal, String> {
    Ok(BigDecimal::from(i))
}

pub fn opti32_to_optu32(i: Option<i32>) -> Result<Option<u32>, String> {
    match i {
        Some(value) => i32_to_u32(value).map(Some),
        None => Ok(None),
    }
}
pub fn optu32_to_opti32(i: Option<u32>) -> Result<Option<i32>, String> {
    match i {
        Some(value) => u32_to_i32(value).map(Some),
        None => Ok(None),
    }
}

pub fn deserialize_distribution(i: Value) -> Result<Vec<UniquesDistributionExtended>, String> {
    serde_json::from_value(i).map_err(|e| e.to_string())
}
pub fn serialize_distribution(i: Vec<UniquesDistributionExtended>) -> Result<Value, String> {
    serde_json::to_value(i).map_err(|e| e.to_string())
}

pub fn deserialize_numbers(i: Value) -> Result<Vec<NiceNumbersExtended>, String> {
    serde_json::from_value(i).map_err(|e| e.to_string())
}
pub fn serialize_numbers(i: Vec<NiceNumbersExtended>) -> Result<Value, String> {
    serde_json::to_value(i).map_err(|e| e.to_string())
}
