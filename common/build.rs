//! Build script for nice_common.
//!
//! When the GPU feature is enabled, this script manages CUDA-related build configuration.
//! The CUDA kernels are embedded in the binary and compiled at runtime via NVRTC.

fn main() {
    #[cfg(feature = "gpu")]
    {
        // The CUDA kernels are embedded in the binary via include_str!()
        // and compiled at runtime using NVRTC, so we don't need to compile
        // them at build time.

        // Tell cargo to rerun this build script if the CUDA kernels change
        println!("cargo:rerun-if-changed=src/cuda/nice_kernels.cu");

        // Note: cudarc with "cuda-version-from-build-system" and "fallback-latest"
        // will try to detect the CUDA version using nvcc at build time.
        // If nvcc is not found (building without CUDA toolkit installed),
        // it will fall back to using the latest CUDA bindings.
        // This allows building on machines without CUDA installed, then
        // running the binary on machines with CUDA GPUs.

        println!(
            "cargo:warning=CUDA kernels will be compiled at runtime via NVRTC so the host MUST have CUDA Toolkit 11.4+ and drivers available for GPU operation."
        );
    }

    #[cfg(not(feature = "gpu"))]
    {
        // Nothing to do when GPU feature is disabled
    }
}
