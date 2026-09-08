//! Build script for nice_common.

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
    // Under the `cuda` feature, register the CUDA kernel source as a
    // rebuild trigger.
    if feature("cuda") {
        println!("cargo:rerun-if-changed=src/cuda/nice_kernels.cu");
    }

    // cubecl-hip links libamdhip64/libhiprtc through cubecl-hip-sys, which
    // finds them by running `hipconfig`. When it can't, it deliberately does
    // not fail (so clippy works without ROCm). Same probe as upstream:
    // `hipconfig` on PATH. Cargo doesn't track PATH, so
    // ROCM_PATH/HIP_PATH (which upstream also honours) are the rerun triggers.
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
                 `hip*` symbols. Install ROCm (https://rocm.docs.amd.com/) to continue.",
            );
        }
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
