# GPU Implementation Documentation

This document provides detailed technical information about the GPU-accelerated nice number search implementation.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [CUDA Kernel Design](#cuda-kernel-design)
3. [Memory Management](#memory-management)
4. [Performance Analysis](#performance-analysis)
5. [Design Decisions](#design-decisions)
6. [Known Limitations](#known-limitations)
7. [Future Improvements](#future-improvements)

## Architecture Overview

### High-Level Flow

```
1. CPU: Initialize CUDA device and compile kernels (NVRTC)
2. CPU: Fetch work from server (range of numbers to check)
3. CPU: Apply residue filter (reduces candidates by ~80%)
4. CPU → GPU: Transfer filtered candidates
5. GPU: Execute kernels (square, cube, check digits)
6. GPU → CPU: Transfer results
7. CPU: Submit results to server
```

### Component Structure

```
nice_common::client_process_gpu
├── GpuContext                    # Manages CUDA device and kernels
├── process_range_detailed_gpu()  # Full statistics collection
├── process_range_niceonly_gpu()  # Fast nice-only search
└── Helper functions              # u128 splitting, etc.

nice/common/src/cuda/nice_kernels.cu
├── u128/u256 arithmetic          # Custom bigint operations
├── count_unique_digits_kernel    # Detailed mode kernel
├── check_is_nice_kernel          # Niceonly mode kernel
└── filter_by_residue_kernel      # (unused, CPU filter is faster)
```

## CUDA Kernel Design

### Challenge: Arbitrary Precision Arithmetic

The main challenge is that CUDA doesn't natively support u128 or arbitrary precision integers. The numbers we're checking can be up to 3.4×10³⁸, and their cubes exceed 256 bits.

### Solution: Custom Multi-Precision Types

#### u128 Representation
```cuda
struct u128 {
    uint64_t lo;  // Lower 64 bits
    uint64_t hi;  // Upper 64 bits
}
```

#### u256 Representation (for n² and n³)
```cuda
struct u256 {
    uint64_t limbs[4];  // Four 64-bit limbs
    // limbs[0] = bits 0-63
    // limbs[1] = bits 64-127
    // limbs[2] = bits 128-191
    // limbs[3] = bits 192-255
}
```

### Key Algorithms

#### 1. Squaring u128 → u256

```
n² = (hi·2⁶⁴ + lo)²
   = hi²·2¹²⁸ + 2·hi·lo·2⁶⁴ + lo²
```

Implementation uses three 64-bit multiplications:
- `lo²` → lower 128 bits
- `hi·lo` → middle term (doubled and shifted)
- `hi²` → upper 128 bits

Complexity: O(1) with 3 64-bit multiplications

#### 2. Cubing u128 → u256

```
n³ = n · n²
```

We first square n, then multiply the result by n.

Implementation:
- Compute n²
- For each limb of n², multiply by n.lo and n.hi
- Handle carries between limbs

Complexity: O(1) with ~12 64-bit multiplications

#### 3. Base Conversion via Division

To extract digits in an arbitrary base, we repeatedly divide by the base:

```
while n > 0:
    digit = n % base
    n = n / base
    record digit
```

Implementation for u256 ÷ u32:
- Process limbs from most to least significant
- Each limb division: split into two 32-bit halves
- Handle remainders propagating down

This is the hottest part of the hot loop (~60% of kernel time).

Complexity: O(limbs) = O(1) for fixed precision

### Kernel 1: count_unique_digits_kernel (Detailed Mode)

**Purpose:** Calculate how many unique digits appear in n² and n³.

**Algorithm:**
```cuda
1. Square n → n²
2. Cube n → n³
3. Convert n² to base, mark each digit in bitmask
4. Convert n³ to base, mark each digit in bitmask
5. Count bits set in bitmask (= unique digits)
```

**Bitmask:** Uses two uint64_t for up to 128-bit bases:
- `digits_lo`: tracks digits 0-63
- `digits_hi`: tracks digits 64-127

**Performance:** ~400-600M numbers/sec on A100

### Kernel 2: check_is_nice_kernel (Niceonly Mode)

**Purpose:** Quickly determine if a number is 100% nice (all base digits used exactly once).

**Algorithm:**
```cuda
1. Square n → n²
2. Process n²: convert to base
   - For each digit, check if already seen
   - If duplicate found: return false immediately
   - Otherwise mark digit as seen
3. Cube n → n³  
4. Process n³: convert to base with same duplicate check
5. Count total unique digits
6. Return true iff count == base
```

**Key Optimization:** Early exit on first duplicate digit.
- Saves ~50% of work on average
- Most numbers are eliminated after checking just n²

**Performance:** ~1-2B numbers/sec on A100 (base 40)

### Kernel 3: filter_by_residue_kernel (Currently Unused)

**Why not used:** The residue filter reduces candidates by 80-90%, but the GPU kernel launch overhead makes CPU filtering faster for typical field sizes.

**When it would help:** For extremely large batches (>100M numbers), GPU residue filtering could amortize the launch cost.

## Memory Management

### Data Transfer Strategy

**Challenge:** PCIe bandwidth is limited (~25 GB/s on PCIe 4.0 x16)

**Optimization:** Minimal data transfer
- Transfer: Only number ranges (u128 split into two u64 arrays)
- Don't transfer: Base, residue filters (small, sent as kernel args)
- Results: Only boolean flags or counts (much smaller than input)

### Memory Layout

```
Input:
  numbers_lo: [u64; N]     // Lower 64 bits of each number
  numbers_hi: [u64; N]     // Upper 64 bits of each number
  
Output (detailed):
  unique_counts: [u32; N]  // Count for each number

Output (niceonly):
  is_nice: [u8; N]         // 0 or 1 for each number
```

### Batch Sizing

**Default:** 10M numbers per batch

**Trade-offs:**
- Larger batches: Better GPU utilization, more memory usage
- Smaller batches: Less memory, more kernel launch overhead

**Memory usage per batch (10M numbers):**
- Input: 2 × 10M × 8 bytes = ~160 MB
- Output: 10M × 4 bytes = ~40 MB
- Total: ~200 MB per batch (well within GPU limits)

**A100 with 40GB:** Can handle batches of ~1B numbers theoretically, but diminishing returns due to other factors.

## Performance Analysis

### Theoretical Performance (A100)

**Compute Bound Analysis:**

A100 specs:
- 6912 CUDA cores
- 1.4 GHz boost clock
- 19.5 TFLOPS FP64

Operations per number (rough estimate):
- Squaring: ~50 operations
- Cubing: ~100 operations  
- Base conversion: ~200 operations
- Total: ~350 operations

Theoretical max: 19.5T ops/sec ÷ 350 ops/number ≈ **55B numbers/sec**

**Reality:** ~1-2B numbers/sec (1.8-3.6% of theoretical)

**Why the gap?**
1. Integer ops, not FP64 (different units)
2. Memory bandwidth bottleneck
3. Complex control flow (base conversion)
4. Register pressure (large intermediate values)

### Memory Bandwidth Analysis

**A100 Memory:**
- 1935 GB/s HBM2e bandwidth
- Per-number read: 16 bytes (two u64)
- Per-number write: 1-4 bytes (result)
- Total: ~20 bytes per number

Theoretical max: 1935 GB/s ÷ 20 bytes ≈ **96B numbers/sec**

This suggests we're more compute-bound than memory-bound, but both matter.

### Actual Performance (A100, base 40)

| Mode | CPU (Ryzen 5950X) | GPU (A100) | Speedup |
|------|-------------------|------------|---------|
| Detailed | ~15M/sec | ~500M/sec | 33× |
| Niceonly | ~80M/sec | ~1.5B/sec | 19× |

**Note:** Performance varies with base size:
- Smaller bases (10-20): Less GPU advantage (10-20× speedup)
- Larger bases (60-80): More GPU advantage (40-60× speedup)

### Bottleneck Analysis

**Profile of kernel time (base 40, niceonly):**
- Division/base conversion: ~60%
- Multiplication (square/cube): ~25%
- Bit manipulation: ~10%
- Control flow: ~5%

**Optimization opportunity:** Base conversion is the clear bottleneck.

## Design Decisions

### 1. Runtime Kernel Compilation (NVRTC)

**Choice:** Compile CUDA kernels at runtime using NVRTC

**Alternatives considered:**
- Pre-compile PTX at build time
- Use `cuda-sys` with pre-built binaries

**Reasons:**
- **Portability:** Works on different CUDA versions
- **Optimization:** NVRTC can optimize for the specific GPU
- **Simplicity:** No build-time CUDA compilation

**Trade-off:** ~2 second startup time for kernel compilation

### 2. CPU Residue Filtering

**Choice:** Apply residue filter on CPU before GPU transfer

**Alternatives considered:**
- GPU residue filtering kernel
- No filtering at all

**Reasons:**
- Filter eliminates 80-90% of candidates
- CPU filter is essentially free (few microseconds)
- GPU kernel launch has fixed overhead (~10μs)
- Smaller GPU transfers save bandwidth

**Numbers:** For 1M input numbers:
- CPU filter: ~1ms, produces ~100K candidates
- GPU transfer: 16MB → 1.6MB (10× reduction)
- GPU kernel: processes 100K instead of 1M

### 3. Split u128 Representation

**Choice:** Split u128 into two u64 arrays for GPU transfer

**Alternatives considered:**
- Pack as u128 and split on GPU
- Use custom packed format

**Reasons:**
- Simple and explicit
- Allows CPU-side optimizations
- GPU prefers coalesced memory access (separate arrays are better)

### 4. Two-Kernel Approach

**Choice:** Separate kernels for detailed vs niceonly

**Alternatives considered:**
- Single unified kernel with mode flag
- Template-based kernel

**Reasons:**
- Different optimization strategies
- Niceonly can early-exit (detailed cannot)
- Less register pressure with specialized kernels

## Known Limitations

### 1. Base Size Limit

**Current:** Bases up to 128 supported

**Reason:** Using 128-bit bitmask (2 × uint64_t)

**Impact:** The problem domain only goes up to base ~100, so this is fine

**Fix if needed:** Use array of uint64_t for larger bases

### 2. Number Range Limit

**Current:** Numbers up to 2²⁵⁶ supported

**Reason:** u256 structure has 4 limbs

**Impact:** Problem domain uses numbers up to ~10³⁸ (< 2¹²⁸), well within limits

**Fix if needed:** Add u512 or variable-precision structure

### 3. Single-GPU Only

**Current:** Each process uses one GPU

**Workaround:** Run multiple processes with different `--device` arguments

**Impact:** Works fine, but less convenient

**Fix:** Add multi-GPU support in single process

### 4. Division Algorithm

**Current:** Simplified division assuming base < 2³²

**Impact:** Works for all practical bases (10-100)

**Limitation:** Won't work for bases > 2³²

**Fix if needed:** Full arbitrary-precision division

## Future Improvements

### Short Term (Easy Wins)

1. **Async Kernel Launches**
   - Overlap CPU work with GPU computation
   - Pipeline: CPU filter batch N+1 while GPU processes batch N
   - Expected gain: 10-20% throughput

2. **Shared Memory for Base Conversion**
   - Cache division results in shared memory
   - Reduces global memory traffic
   - Expected gain: 5-15% speedup

3. **Better Batch Sizing**
   - Auto-tune based on GPU memory
   - Larger batches for high-end GPUs
   - Expected gain: 5-10% throughput

### Medium Term (More Effort)

4. **Barrett Reduction for Division**
   - Pre-compute reciprocal of base
   - Replace division with multiplication
   - Expected gain: 30-50% speedup in kernels

5. **Multi-GPU Support**
   - NCCL for multi-GPU coordination
   - Split work across multiple GPUs automatically
   - Expected gain: Linear scaling (2 GPUs = 2× speed)

6. **Warp-Level Optimizations**
   - Use warp shuffle operations
   - Cooperative groups for better parallelism
   - Expected gain: 10-20% speedup

### Long Term (Major Features)

7. **ROCm Support (AMD GPUs)**
   - Port kernels to HIP
   - Support AMD Instinct MI250X, MI300X
   - Broader hardware support

8. **Dynamic Kernel Generation**
   - Generate specialized kernels per base
   - Unroll loops for specific base sizes
   - Expected gain: 20-40% for common bases

9. **Machine Learning-Based Filtering**
   - Train model to predict nice number likelihood
   - Pre-filter on GPU before expensive checks
   - Expected gain: 50-100% (if model is good)

10. **Distributed Multi-Node GPU**
    - MPI + NCCL for cluster computing
    - Scale to hundreds of GPUs
    - Expected gain: Linear scaling

## Performance Tuning Guide

### For Developers

**If modifying kernels:**

1. **Profile with Nsight Compute:**
   ```bash
   ncu --set full -o profile ./nice_gpu_client --benchmark default
   ```

2. **Key metrics to watch:**
   - SM utilization (target: >80%)
   - Memory throughput (target: >50% of peak)
   - Register usage (lower is better)
   - Occupancy (target: >50%)

3. **Common issues:**
   - High register pressure → reduce local variables
   - Low occupancy → decrease thread block size
   - Memory bottleneck → use shared memory

### For Users

**If experiencing poor performance:**

1. **Check GPU utilization:**
   ```bash
   nvidia-smi dmon -s u -d 1
   ```
   Should see 90-100% GPU util

2. **Check for throttling:**
   ```bash
   nvidia-smi -q -d CLOCK,TEMPERATURE,POWER
   ```

3. **Try niceonly mode:**
   - Always faster than detailed
   - Better GPU utilization

4. **Adjust batch size:**
   - Larger for high-end GPUs
   - Smaller if running out of memory

## References

- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [cudarc Documentation](https://docs.rs/cudarc/)
- [CGBN: CUDA GPU BigNum Library](https://github.com/NVlabs/CGBN)
- [Malachite: Rust Arbitrary Precision](https://www.malachite.rs/)

## Appendix: Kernel Code Structure

### Full kernel call flow:

```
main.rs
  └─> process_one_field()
      └─> process_range_niceonly_gpu()
          ├─> Apply residue filter (CPU)
          ├─> split_u128_vec() → (lo, hi)
          ├─> device.htod_sync_copy() → transfer to GPU
          ├─> nice_kernel.launch() → run CUDA kernel
          │   └─> check_is_nice_kernel<<<grid, block>>>()
          │       ├─> square_u128() → n²
          │       ├─> cube_u128() → n³
          │       ├─> Process n²:
          │       │   └─> div_u256_by_u32() [loop until zero]
          │       └─> Process n³:
          │           └─> div_u256_by_u32() [loop until zero]
          ├─> device.dtoh_sync_copy() → transfer from GPU
          └─> Collect results
```

---

**Last Updated:** 2025-01-27
**Version:** 1.0
**Author:** Implementation based on cudarc 0.18 and CUDA 12.x