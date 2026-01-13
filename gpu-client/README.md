# Nice GPU Client

GPU-accelerated client for distributed search of square-cube pandigitals using CUDA.

## Overview

This is a CUDA-accelerated version of the nice client that runs computations on NVIDIA GPUs. It's designed to take advantage of the massive parallelism offered by GPUs like the A100 to search for nice numbers significantly faster than the CPU-only client.

**Expected Performance:** 10-50x speedup over the CPU client, depending on GPU model and base size.

## Requirements

### Hardware
- NVIDIA GPU with CUDA support
- Compute Capability 3.5 or higher recommended
- Tested on: A100, RTX 3090, RTX 4090

### Software
- NVIDIA GPU drivers (525.x or later recommended)
- CUDA Toolkit 11.4 or later (for compilation)
  - The toolkit includes `nvcc` which is required
- Rust toolchain (1.70+)

## Installation

### 1. Install CUDA Toolkit

**Ubuntu/Debian:**
```bash
# Add NVIDIA package repository
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update

# Install CUDA toolkit
sudo apt-get install cuda-toolkit-12-3
```

**Verify installation:**
```bash
nvcc --version
nvidia-smi
```

### 2. Build the GPU Client

```bash
# Clone the repository
git clone https://github.com/wasabipesto/nice.git
cd nice

# Build the GPU client (release mode for best performance)
cargo build --release -p nice_gpu_client

# The binary will be at: target/release/nice_gpu_client
```

## Usage

### Basic Usage

```bash
# Run once with default settings
./nice_gpu_client

# Run with a specific username
./nice_gpu_client --username yourname

# Run forever
./nice_gpu_client --repeat

# Use a specific GPU (for multi-GPU systems)
./nice_gpu_client --device 1
```

### Command Line Options

```
Usage: nice_gpu_client [OPTIONS] [MODE]

Arguments:
  [MODE]
          The checkout mode to use
          
          Possible values:
          - detailed: Get detailed stats on all numbers
          - niceonly: Optimized search for 100% nice numbers only (faster)
          
          [default: detailed]

Options:
      --api-base <API_BASE>
          The base API URL to connect to
          [default: https://api.nicenumbers.net]

  -u, --username <USERNAME>
          The username to send alongside your contribution
          [default: anonymous]

  -r, --repeat
          Run indefinitely with the current settings

  -q, --quiet
          Suppress all output

  -v, --verbose
          Show additional output

  -d, --device <DEVICE>
          CUDA device to use (0 for first GPU, 1 for second, etc.)
          [default: 0]

  -b, --benchmark <BENCHMARK>
          Run an offline benchmark
          
          Possible values:
          - default:     1e6 @ base 40
          - large:       1e8 @ base 40
          - extra-large: 1e9 @ base 40
          - hi-base:     1e6 @ base 80

  -h, --help
          Print help

  -V, --version
          Print version
```

### Environment Variables

All options can also be set via environment variables:

```bash
export NICE_MODE=niceonly
export NICE_USERNAME=myusername
export NICE_GPU_DEVICE=0
export NICE_REPEAT=1

./nice_gpu_client
```

### Benchmarking

Compare GPU vs CPU performance:

```bash
# GPU benchmark
./nice_gpu_client --benchmark extra-large

# CPU benchmark (for comparison)
cd ../target/release
./nice_client --benchmark extra-large
```

## Performance Tips

1. **Use niceonly mode** for maximum throughput (~20x faster than detailed)
2. **Multiple GPUs?** Run multiple instances with different `--device` values
3. **Thermal throttling?** Monitor with `nvidia-smi` and ensure adequate cooling
4. **Power limits?** A100s can be power-limited; check with `nvidia-smi -q -d POWER`

## Troubleshooting

### "Failed to initialize GPU"

**Check GPU is visible:**
```bash
nvidia-smi
```

**Check CUDA is installed:**
```bash
nvcc --version
```

**Check driver version:**
```bash
cat /proc/driver/nvidia/version
```

### "Failed to compile CUDA kernels"

This means NVRTC couldn't compile the kernels at runtime.

**Verify CUDA toolkit is fully installed:**
```bash
which nvcc
ls -la /usr/local/cuda/
```

**Check CUDA version compatibility:**
- CUDA 11.4+ is required
- If you have multiple CUDA versions, set `CUDA_HOME`:
  ```bash
  export CUDA_HOME=/usr/local/cuda-12.3
  export PATH=$CUDA_HOME/bin:$PATH
  export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
  ```

### "Out of memory" errors

The GPU has limited memory. Try:
1. Reducing batch size (default is 10M numbers)
2. Using niceonly mode (less memory overhead)
3. Closing other GPU applications

### Poor performance

1. **Check GPU utilization:** `nvidia-smi dmon`
   - Should see 90-100% GPU utilization
2. **Thermal throttling?** Check temperature and clock speeds
3. **PCIe bottleneck?** Ensure GPU is in a PCIe 3.0 x16 or better slot

## Architecture

The GPU client uses:
- **CUDA kernels** written in C++ for the hot loops (squaring, cubing, base conversion)
- **cudarc** for safe Rust bindings to CUDA
- **NVRTC** for runtime kernel compilation
- **u128 arithmetic** split into u64 pairs for GPU compatibility

Key optimizations:
- Early exit on duplicate digits (niceonly mode)
- Residue filtering on CPU before GPU processing
- Batch processing to amortize transfer costs
- Shared memory for residue filters

## Development

### Running Tests

```bash
# Run GPU tests (requires GPU)
cargo test -p nice_common --features gpu

# Run GPU client tests
cargo test -p nice_gpu_client
```

### Modifying CUDA Kernels

The kernels are in `common/src/cuda/nice_kernels.cu`. After editing:

1. Rebuild: `cargo build -p nice_gpu_client`
2. The kernels are compiled at runtime via NVRTC
3. Test with: `./nice_gpu_client --benchmark default`

## Comparison with CPU Client

| Feature | CPU Client | GPU Client |
|---------|-----------|------------|
| Speed (detailed) | 10-20M/sec | 100-500M/sec |
| Speed (niceonly) | 50-100M/sec | 500M-2B/sec |
| Memory | ~100MB | ~2-4GB VRAM |
| Dependencies | None | CUDA Toolkit |
| Portability | All platforms | NVIDIA GPUs only |

## Contributing

Found a bug or want to improve performance? PRs welcome!

Areas for improvement:
- Support for multi-GPU (single process)
- Further kernel optimizations
- ROCm support for AMD GPUs
- Async kernel launches

## License

Same as the main project: MIT OR Apache-2.0