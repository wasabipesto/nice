//! Build script for nice_common.
//!
//! Two jobs. Under the `cuda` feature, register the CUDA kernel source as a
//! rebuild trigger: the kernels are embedded with `include_str!` and compiled
//! at runtime by NVRTC, so nothing here needs the toolkit and the build works
//! on any machine.
//!
//! And for the feature combinations that build fine but then fail, or do
//! nothing, in ways whose output never names the cause, say so up front.
//! Warnings from a workspace member's build script are always displayed,
//! which is the whole point: the same warning from a registry crate (see
//! `cubecl-hip-sys`) is suppressed. Warn, never fail — `cargo clippy
//! --features cubecl-hip` on a machine without ROCm has to keep working,
//! which is the same reason upstream doesn't fail either.

use std::env;

/// Whether a Cargo feature of this crate is enabled for this build.
fn feature(name: &str) -> bool {
    let key = format!("CARGO_FEATURE_{}", name.to_uppercase().replace('-', "_"));
    env::var_os(key).is_some()
}

/// One-line warning shown in the build output (newlines would be dropped).
fn warn(msg: &str) {
    println!("cargo:warning={msg}");
}

fn main() {
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let macos = target_os == "macos";

    if feature("cuda") {
        println!("cargo:rerun-if-changed=src/cuda/nice_kernels.cu");
    }

    // cubecl-hip links libamdhip64/libhiprtc through cubecl-hip-sys, which
    // finds them by running `hipconfig`. When it can't, it deliberately does
    // not fail (so clippy works without ROCm): it warns, and as a registry
    // crate that warning is never shown. The build then dies at the final
    // link with a wall of `undefined symbol: hipSetDevice` and friends that
    // mentions neither HIP nor ROCm. Same probe as upstream: `hipconfig` on
    // PATH. Cargo doesn't track PATH, so ROCM_PATH/HIP_PATH (which upstream
    // also honours) are the rerun triggers.
    println!("cargo:rerun-if-env-changed=ROCM_PATH");
    println!("cargo:rerun-if-env-changed=HIP_PATH");
    if feature("cubecl-hip") {
        let hipconfig = std::process::Command::new("hipconfig")
            .arg("--version")
            .output()
            .is_ok();
        if !hipconfig {
            warn(
                "cubecl-hip: `hipconfig` is not on PATH, so ROCm is not installed or not \
                 visible to this build. The build will fail at the final link with undefined \
                 `hip*` symbols. Install ROCm (https://rocm.docs.amd.com/) or drop the \
                 cubecl-hip feature; `cargo check` and `cargo clippy` still work without it.",
            );
        }
    }

    // cubecl-spirv: CubeCL's SPIR-V compiler only engages when wgpu is on its
    // Vulkan backend, and on macOS cubecl-wgpu's AutoGraphicsApi always picks
    // Metal, so the feature is inert there. It also can't build on macOS
    // without VULKAN_SDK set: cubecl-wgpu's own build script panics. That
    // script runs before this one (it's our dependency), so this warning is
    // only seen when VULKAN_SDK *is* set and the build gets this far.
    println!("cargo:rerun-if-env-changed=VULKAN_SDK");
    if feature("cubecl-spirv") && macos {
        warn(
            "cubecl-spirv has no effect on macOS: the cubecl backend always runs over Metal \
             there, so the SPIR-V compiler never engages. Build without it, or with \
             cubecl-metal for CubeCL's direct MSL compiler.",
        );
    }

    // cubecl-metal: cubecl-wgpu gates its MSL compiler on
    // `all(feature = "msl", target_os = "macos")`, so anywhere else the
    // feature compiles and does nothing.
    if feature("cubecl-metal") && !macos {
        warn(&format!(
            "cubecl-metal has no effect on {target_os}: cubecl-wgpu only enables its MSL \
             compiler on macOS. On Vulkan devices cubecl-spirv is the direct compiler."
        ));
    }

    // openssl-tls + rustls-tls together: reqwest compiles both stacks and its
    // TlsBackend::default() picks native-tls. This happens whenever
    // openssl-tls is added without --no-default-features, and legitimately in
    // workspace builds that unify api's rustls-tls with a client openssl-tls,
    // which is why it is a warning and not a compile_error!.
    if feature("openssl-tls") && feature("rustls-tls") {
        warn(
            "both openssl-tls and rustls-tls are enabled: both TLS stacks are compiled in and \
             reqwest will use native-tls (OpenSSL). For OpenSSL only, build with \
             --no-default-features --features openssl-tls.",
        );
    }
}
