//! WebAssembly interface for nice number processing with Web Worker support
//!
//! This module provides a browser-compatible client for the distributed computing
//! project that finds "nice numbers" (square-cube pandigitals).

use nice_common::FieldSize;
use nice_common::detailed_fd::process_range_detailed_fd;
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

/// The workspace client version, so browser submissions carry the real
/// number instead of a string hardcoded in the JS (which had drifted three
/// minor versions behind). The workers append a `-wasm-worker` /
/// `-wasm-webgpu` suffix to keep the paths distinguishable server-side.
#[wasm_bindgen]
pub fn client_version() -> String {
    nice_common::CLIENT_VERSION.to_string()
}

/// The git commit the wasm module was built from, for benchmark reports.
#[wasm_bindgen]
pub fn build_sha() -> String {
    nice_common::BUILD_SHA.to_string()
}

/// Process a chunk of numbers and return nice numbers and distribution updates.
///
/// Goes through the same CPU detailed dispatch as the native client: finite
/// differences at the bases that have them, the division loop elsewhere.
#[wasm_bindgen]
pub fn process_chunk_wasm(range_start_str: &str, range_end_str: &str, base: u32) -> String {
    console_error_panic_hook::set_once();

    // Get range start and end
    let range_start = u128::from_str(range_start_str).unwrap();
    let range_end = u128::from_str(range_end_str).unwrap();
    let range = FieldSize::new(range_start, range_end);

    // Pass off to common for processing
    let result = process_range_detailed_fd(&range, base);

    // Send results back to worker
    serde_json::to_string(&result).unwrap()
}

// ============================================================================
// Benchmark suite
// ============================================================================

/// The shared benchmark plan, resolved for the browser: the detailed
/// scenarios from `nice_common::bench_defs` (the wasm client has no niceonly
/// path), with start positions resolved against the base ranges. u128 values
/// are carried as strings for the same reason the claim ranges are.
///
/// The page drives the sweep from this instead of hardcoding windows, so the
/// browser suite and the native `--benchmark` sweep measure identical work
/// by construction.
#[wasm_bindgen]
pub fn benchmark_plan() -> String {
    let scenarios: Vec<serde_json::Value> = nice_common::bench_defs::DETAILED_SCENARIOS
        .iter()
        .map(|def| {
            serde_json::json!({
                "key": def.key,
                "base": def.base,
                "start": def.resolved_start().to_string(),
                "window_cpu": def.window_cpu.to_string(),
                "window_gpu": def.window_gpu.to_string(),
                "character": def.character,
                "single_thread": def.single_thread,
            })
        })
        .collect();
    serde_json::json!({
        "schema_version": nice_common::bench_defs::BENCH_SCHEMA_VERSION,
        "scenarios": scenarios,
    })
    .to_string()
}

/// The synthetic NiceMark score for a set of measured rates, via the same
/// geometric mean over the same pinned references the native client scores
/// against — a browser and a native run on one machine are directly
/// comparable (the browser just scores lower). Input:
/// `[{"key": "b40_detailed", "rate": 1.5e7}, ...]`. Returns nothing when no
/// scenario matched a reference.
#[wasm_bindgen]
pub fn nicemark_score(rates_json: &str, gpu: bool) -> Result<Option<f64>, JsValue> {
    let rates: serde_json::Value = serde_json::from_str(rates_json)
        .map_err(|e| JsValue::from_str(&format!("bad rates: {e}")))?;
    let pairs: Vec<(&str, f64)> = rates
        .as_array()
        .ok_or_else(|| JsValue::from_str("rates must be an array"))?
        .iter()
        .filter_map(|entry| Some((entry.get("key")?.as_str()?, entry.get("rate")?.as_f64()?)))
        .collect();
    Ok(nice_common::bench_defs::compute_score(
        pairs.iter().copied(),
        gpu,
    ))
}

// ============================================================================
// GPU (WebGPU via CubeCL)
// ============================================================================

use nice_common::cubecl_backend::CubeclContext;
use nice_common::cubecl_web::{
    CUBECL_WEB_BATCH_SIZE, WebKernel, fd_seed_probe_web_async, process_range_detailed_web_async,
    process_range_detailed_web_kernel,
};

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

/// Process a chunk on the GPU with the u32-only WebGPU kernels (finite
/// differences at bases up to 64). Same JSON result shape as
/// [`process_chunk_wasm`], so the merging JS is shared.
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

/// Test hook: [`process_chunk_gpu`] with the run shape chosen. `run_len` 0
/// runs the per-number kernel on every candidate; otherwise the
/// finite-difference kernel takes the plan's interval with `run_len`
/// candidates per thread and at most `batch` per dispatch (0 for the
/// production batch). For the browser parity and tuning harness only.
#[doc(hidden)]
#[wasm_bindgen]
pub async fn process_chunk_gpu_shape(
    range_start_str: &str,
    range_end_str: &str,
    base: u32,
    run_len: u32,
    batch: u32,
) -> Result<String, JsValue> {
    let range_start = u128::from_str(range_start_str)
        .map_err(|e| JsValue::from_str(&format!("bad range_start: {e}")))?;
    let range_end = u128::from_str(range_end_str)
        .map_err(|e| JsValue::from_str(&format!("bad range_end: {e}")))?;
    let range = FieldSize::new(range_start, range_end);
    let kernel = if run_len == 0 {
        WebKernel::Division
    } else {
        WebKernel::FiniteDifference {
            run_len,
            batch: if batch == 0 {
                CUBECL_WEB_BATCH_SIZE
            } else {
                u128::from(batch)
            },
        }
    };
    let ctx = CubeclContext::new_default_async()
        .await
        .map_err(|e| JsValue::from_str(&format!("WebGPU unavailable: {e:#}")))?;
    let result = process_range_detailed_web_kernel(&ctx, &range, base, kernel)
        .await
        .map_err(|e| JsValue::from_str(&format!("GPU processing failed: {e:#}")))?;
    serde_json::to_string(&result).map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Test hook: the GPU finite-difference seed for `count` consecutive n from
/// `start` at `base`, as JSON: the record layout (`stride` words per n,
/// `offsets` of S, C, dS, dC and ddC) and the flat `words`, each holding
/// four base-`base` digits, least significant first. Words past an array's
/// top are 0. For the browser parity harness only.
#[doc(hidden)]
#[wasm_bindgen]
pub async fn fd_seed_probe_gpu(start_str: &str, count: u32, base: u32) -> Result<String, JsValue> {
    let start =
        u128::from_str(start_str).map_err(|e| JsValue::from_str(&format!("bad start: {e}")))?;
    let ctx = CubeclContext::new_default_async()
        .await
        .map_err(|e| JsValue::from_str(&format!("WebGPU unavailable: {e:#}")))?;
    let words = fd_seed_probe_web_async(&ctx, base, start, count)
        .await
        .map_err(|e| JsValue::from_str(&format!("seed probe failed: {e:#}")))?;
    Ok(serde_json::json!({
        "stride": nice_common::cubecl_web::FD_SEED_STRIDE,
        "offsets": nice_common::cubecl_web::FD_SEED_OFFSETS,
        "words": words,
    })
    .to_string())
}

/// Sizes the GPU path actually compiled with, for field diagnosis. The page
/// logs this at startup: a browser caches the wasm aggressively, and without
/// it a console paste cannot tell a fixed build from a stale one.
#[wasm_bindgen]
pub fn gpu_build_info() -> String {
    format!(
        "{{\"near_miss_capacity\":{},\"batch_size\":{},\"miss_buffer_bytes\":{},\"detailed_kernel\":\"fd\"}}",
        nice_common::cubecl_web::near_miss_capacity(),
        CUBECL_WEB_BATCH_SIZE,
        nice_common::cubecl_web::near_miss_capacity() * 5 * 4,
    )
}
