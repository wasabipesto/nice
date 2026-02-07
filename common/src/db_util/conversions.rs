//! Safe-ish conversions between rust and sql types.
//! Ideally this will be ripped out and implemented as whatever custom diesel types end up being necessary.

#![allow(
    clippy::unnecessary_wraps,
    clippy::needless_pass_by_value,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]

use super::*;

pub fn i64_to_u128(i: i64) -> Result<u128> {
    if i < 0 {
        bail!("i64 value is negative and cannot be converted to u128")
    }
    Ok(i as u128)
}
pub fn u128_to_i64(i: u128) -> Result<i64> {
    if i > i64::MAX as u128 {
        bail!("u128 value exceeds i64::MAX and cannot be converted to i64")
    }
    Ok(i as i64)
}

pub fn i32_to_u128(i: i32) -> Result<u128> {
    if i < 0 {
        bail!("i32 value is negative and cannot be converted to u128")
    }
    Ok(i as u128)
}
pub fn u128_to_i32(i: u128) -> Result<i32> {
    if i > i32::MAX as u128 {
        bail!("u128 value exceeds i32::MAX and cannot be converted to i32")
    }
    Ok(i as i32)
}

pub fn i32_to_u32(i: i32) -> Result<u32> {
    if i < 0 {
        bail!("i32 value is negative and cannot be converted to u32")
    }
    Ok(i as u32)
}
pub fn u32_to_i32(i: u32) -> Result<i32> {
    if i > i32::MAX as u32 {
        bail!("u32 value exceeds i32::MAX and cannot be converted to i32")
    }
    Ok(i as i32)
}

pub fn i32_to_u8(i: i32) -> Result<u8> {
    if i < 0 || i > i32::from(u8::MAX) {
        bail!("i32 value is out of range for u8")
    }
    Ok(i as u8)
}
pub fn u8_to_i32(i: u8) -> Result<i32> {
    Ok(i32::from(i))
}

pub fn bigdec_to_u128(i: BigDecimal) -> Result<u128> {
    match i.to_u128() {
        Some(value) => Ok(value),
        None => bail!("BigDecimal value cannot be converted to u128"),
    }
}
pub fn u128_to_bigdec(i: u128) -> Result<BigDecimal> {
    Ok(BigDecimal::from(i))
}
pub fn f32_to_bigdec(i: f32) -> Result<BigDecimal> {
    BigDecimal::from_f32(i).ok_or_else(|| anyhow!("Failed to convert f32 to BigDecimal"))
}

pub fn opti32_to_optu32(i: Option<i32>) -> Result<Option<u32>> {
    match i {
        Some(value) => i32_to_u32(value).map(Some),
        None => Ok(None),
    }
}
pub fn optu32_to_opti32(i: Option<u32>) -> Result<Option<i32>> {
    match i {
        Some(value) => u32_to_i32(value).map(Some),
        None => Ok(None),
    }
}

pub fn deserialize_distribution(i: Value) -> Result<Vec<UniquesDistribution>> {
    Ok(serde_json::from_value(i)?)
}
pub fn serialize_distribution(i: Vec<UniquesDistribution>) -> Result<Value> {
    Ok(serde_json::to_value(i)?)
}

pub fn deserialize_opt_distribution(i: Option<Value>) -> Result<Option<Vec<UniquesDistribution>>> {
    match i {
        Some(i) => Ok(Some(deserialize_distribution(i)?)),
        None => Ok(None),
    }
}
pub fn serialize_opt_distribution(i: Option<Vec<UniquesDistribution>>) -> Result<Option<Value>> {
    match i {
        Some(i) => Ok(Some(serialize_distribution(i)?)),
        None => Ok(None),
    }
}

pub fn deserialize_numbers(i: Value) -> Result<Vec<NiceNumber>> {
    Ok(serde_json::from_value(i)?)
}
pub fn serialize_numbers(i: Vec<NiceNumber>) -> Result<Value> {
    Ok(serde_json::to_value(i)?)
}

pub fn deserialize_searchmode(i: String) -> Result<SearchMode> {
    match i.as_str() {
        "detailed" => Ok(SearchMode::Detailed),
        "niceonly" => Ok(SearchMode::Niceonly),
        _ => bail!("Failed to deserialize search mode: {i}"),
    }
}
#[must_use]
pub fn serialize_searchmode(i: SearchMode) -> String {
    match i {
        SearchMode::Detailed => "detailed".into(),
        SearchMode::Niceonly => "niceonly".into(),
    }
}
