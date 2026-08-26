//! WebAssembly interface for nice number processing with Web Worker support
//!
//! This module provides a browser-compatible client for the distributed computing
//! project that finds "nice numbers" (square-cube pandigitals).

use nice_common::FieldSize;
use nice_common::client_process::process_range_detailed;
use std::str::FromStr;
use wasm_bindgen::prelude::*;

// Define the panic hook for better error messages in the browser
#[wasm_bindgen(start)]
pub fn main() {
    console_error_panic_hook::set_once();
}

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

/// Process a chunk of numbers and return nice numbers and distribution updates
#[wasm_bindgen]
pub fn process_chunk_wasm(range_start_str: &str, range_end_str: &str, base: u32) -> String {
    console_error_panic_hook::set_once();

    // Get range start and end
    let range_start = u128::from_str(range_start_str).unwrap();
    let range_end = u128::from_str(range_end_str).unwrap();
    let range = FieldSize::new(range_start, range_end);

    // Pass off to common for processing
    let result = process_range_detailed(&range, base);

    // Send results back to worker
    serde_json::to_string(&result).unwrap()
}

// ============================================================================
// GPU (WebGPU via CubeCL)
// ============================================================================

use nice_common::cubecl_backend::CubeclContext;
use nice_common::cubecl_web::process_range_detailed_web_async;

/// Try to bring up WebGPU. Returns the adapter's name (for the backend
/// dropdown) or throws if no adapter is available — the caller treats a
/// rejection as "no GPU option on this browser".
///
/// The underlying client is initialized once per worker and cheaply cloned
/// by later calls, so this doubles as the warm-up.
#[wasm_bindgen]
pub async fn gpu_init() -> Result<String, JsValue> {
    console_error_panic_hook::set_once();
    let ctx = CubeclContext::new_default_async()
        .await
        .map_err(|e| JsValue::from_str(&format!("WebGPU init failed: {e:#}")))?;
    Ok(ctx.device_name())
}

/// Process a chunk on the GPU with the u32-only WebGPU kernel. Same JSON
/// result shape as [`process_chunk_wasm`], so the merging JS is shared.
#[wasm_bindgen]
pub async fn process_chunk_gpu(
    range_start_str: &str,
    range_end_str: &str,
    base: u32,
) -> Result<String, JsValue> {
    let range_start = u128::from_str(range_start_str)
        .map_err(|e| JsValue::from_str(&format!("bad range_start: {e}")))?;
    let range_end = u128::from_str(range_end_str)
        .map_err(|e| JsValue::from_str(&format!("bad range_end: {e}")))?;
    let range = FieldSize::new(range_start, range_end);

    // Cheap after gpu_init: clones the process-wide client.
    let ctx = CubeclContext::new_default_async()
        .await
        .map_err(|e| JsValue::from_str(&format!("WebGPU unavailable: {e:#}")))?;
    let result = process_range_detailed_web_async(&ctx, &range, base)
        .await
        .map_err(|e| JsValue::from_str(&format!("GPU processing failed: {e:#}")))?;
    serde_json::to_string(&result).map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Sizes the GPU path actually compiled with, for field diagnosis. The page
/// logs this at startup: a browser caches the wasm aggressively, and without
/// it a console paste cannot tell a fixed build from a stale one.
#[wasm_bindgen]
pub fn gpu_build_info() -> String {
    format!(
        "{{\"near_miss_capacity\":{},\"batch_size\":{},\"miss_buffer_bytes\":{}}}",
        nice_common::cubecl_web::near_miss_capacity(),
        nice_common::cubecl_web::CUBECL_WEB_BATCH_SIZE,
        nice_common::cubecl_web::near_miss_capacity() * 5 * 4,
    )
}
