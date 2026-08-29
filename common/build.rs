//! Build script for nice_common.
//!
//! When the GPU feature is enabled, this script manages CUDA-related build configuration.
//! The CUDA kernels are embedded in the binary and compiled at runtime via NVRTC.

fn main() {
    #[cfg(feature = "cuda")]
    {
        // The CUDA kernels are embedded in the binary via include_str!()
        // and compiled at runtime using NVRTC, so we don't need to compile
        // them at build time.

        // Tell cargo to rerun this build script if the CUDA kernels change
        println!("cargo:rerun-if-changed=src/cuda/nice_kernels.cu");

        // Nothing here needs the CUDA toolkit: cudarc dlopens the driver and
        // NVRTC at runtime, so the build works on any machine. The runtime
        // requirement (toolkit for the cuda/cubecl-cuda backends, driver only
        // for cubecl/wgpu) is documented on --gpu-backend and in the README.
    }

    #[cfg(not(feature = "cuda"))]
    {
        // Nothing to do when GPU feature is disabled
    }
}
