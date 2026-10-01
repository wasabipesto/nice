//! Build script for nice_common.

use std::env;
use std::path::PathBuf;
use std::process::Command;

/// Whether a Cargo feature of this crate is enabled for this build.
fn feature(name: &str) -> bool {
    let key = format!("CARGO_FEATURE_{}", name.to_uppercase().replace('-', "_"));
    env::var_os(key).is_some()
}

/// One-line warning shown in the build output (newlines would be dropped).
fn warn(msg: &str) {
    println!("cargo:warning={msg}");
}

/// Run `git` in the checkout containing this crate and return trimmed stdout
/// on success. Any failure (no git on PATH, no repository, dubious-ownership
/// refusal inside a container) is `None`.
fn git(args: &[&str]) -> Option<String> {
    let out = Command::new("git")
        .args(args)
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8(out.stdout).ok()?;
    let s = s.trim();
    (!s.is_empty()).then(|| s.to_string())
}

/// The commit this build comes from, as `NICE_BUILD_SHA` for `env!`.
///
/// Precedence: the `NICE_BUILD_SHA` environment variable (CI sets it from
/// `github.sha`; docker builds pass it as a build-arg, since `.git` is not
/// in the docker context), then `git rev-parse HEAD` on the checkout, then
/// `unknown`. The value is the commit HEAD pointed at when the build script
/// ran; it does not track uncommitted changes, because a build script only
/// reruns when its declared inputs change and a `-dirty` mark would go stale.
fn build_sha() {
    println!("cargo:rerun-if-env-changed=NICE_BUILD_SHA");
    let from_env = env::var("NICE_BUILD_SHA")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());
    let sha = match from_env {
        Some(sha) => sha,
        None => {
            // Rerun when HEAD moves: watch HEAD itself and, when it is
            // symbolic, the ref it points at. `--git-path` resolves both
            // through worktrees. A missing ref file (packed refs, detached
            // HEAD) is fine: cargo treats a missing watched path as changed
            // only when it later appears.
            if let Some(head) = git(&["rev-parse", "--git-path", "HEAD"]) {
                println!("cargo:rerun-if-changed={}", to_abs(&head));
            }
            if let Some(r) = git(&["symbolic-ref", "-q", "HEAD"])
                && let Some(path) = git(&["rev-parse", "--git-path", &r])
            {
                println!("cargo:rerun-if-changed={}", to_abs(&path));
            }
            git(&["rev-parse", "HEAD"]).unwrap_or_else(|| "unknown".to_string())
        }
    };
    println!("cargo:rustc-env=NICE_BUILD_SHA={sha}");
}

/// `git rev-parse --git-path` answers relative to the cwd it ran in.
fn to_abs(path: &str) -> String {
    let p = PathBuf::from(path);
    if p.is_absolute() {
        return path.to_string();
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join(p)
        .display()
        .to_string()
}

fn main() {
    build_sha();

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
