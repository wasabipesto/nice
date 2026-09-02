# Nice!

> Join the distributed search for square-cube pandigitals!

## Why does this exist

Square-cube pandigials ("nice" numbers) seem to be distributed pseudo-randomly. It doesn't take very long to check if a number is pandigital in a specific base, but even after we narrow the search range to numbers with the right amount of digits in their square and cube there's a lot of numbers to check. This system coordinates multiple clients to search more efficiently.

For more background, check out the [original article](https://beautifulthorns.wixsite.com/home/post/is-69-unique) and [my findings](https://nicenumbers.net).

## Client Quickstart

The easiest way to get started is by going to [https://nicenumbers.net/search/](https://nicenumbers.net/search) and running it in your browser. You'll see live results and everything will be submitted in your name.

If you want to go even faster, you can run the [native binaries from the latest release](https://github.com/wasabipesto/nice/releases) or run the docker image. We usually see a ~2x speedup versus the browser.

```sh
# Run the release binary
./nice_client

# Run the docker image
docker run -it --init ghcr.io/wasabipesto/nice_client:3

# Run with a username
./nice_client --username gilgamesh

# Run with 12 threads
./nice_client --threads 12

# Run forever
./nice_client --repeat

# The docker image supports these options too!
docker run -it --init ghcr.io/wasabipesto/nice_client:3 --repeat

# Both versions also support environment variables
docker run -it --init -e NICE_USERNAME=gilgamesh ghcr.io/wasabipesto/nice_client:3
```

You may get slightly more performance by building the binaries yourself. Building the client requires rust and a few other dependencies.

```sh
# Install rust and cargo
sudo apt install build-essential curl git
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Clone this repository
git clone https://github.com/wasabipesto/nice.git
cd nice

# Build the client binary
cargo build -r -p nice_client
cd target/release

# Run once with default settings
./nice_client
```

You can find various settings and their options with the `--help` flag:

```
a client for distributed search of square-cube pandigitals

Usage: nice_client [OPTIONS] [MODE]

Arguments:
  [MODE]
          The checkout mode to use

          Possible values:
          - detailed: Get detailed stats on all numbers, important for long-term analytics
          - niceonly: Implements optimizations to speed up the search, usually by a factor of around 20. Does not keep statistics and cannot be quickly verified
          
          [env: NICE_MODE=]
          [default: detailed]

Options:
      --api-base <API_BASE>
          The base API URL to connect to
          
          [env: NICE_API_BASE=]
          [default: https://api.nicenumbers.net]

      --api-max-retries <API_MAX_RETRIES>
          If an API call encounters a retryable error, retry with exponential backoff this many times
          
          [env: NICE_API_MAX_RETRIES=]
          [default: 10]

  -u, --username <USERNAME>
          The username to send alongside your contribution
          
          [env: NICE_USERNAME=]
          [default: anonymous]

  -r, --repeat
          Run indefinitely with the current settings
          
          [env: NICE_REPEAT=]

  -n, --no-progress
          Hide the progress bar
          
          [env: NICE_NO_PROGRESS=]

  -t, --threads <THREADS>
          Run parallel with this many threads
          
          [env: NICE_THREADS=]
          [default: 4]

      --prefetch-seconds <PREFETCH_SECONDS>
          Keep roughly this many seconds of work claimed ahead of the processor. Set to 0 to force the old single-field prefetch
          
          [env: NICE_PREFETCH_SECONDS=]
          [default: 2]

      --prefetch-max <PREFETCH_MAX>
          Never hold more than this many claimed fields at once
          
          [env: NICE_PREFETCH_MAX=]
          [default: 16]

      --prefetch-concurrency <PREFETCH_CONCURRENCY>
          Allow this many claim requests to be in flight at once
          
          [env: NICE_PREFETCH_CONCURRENCY=]
          [default: 4]

  -b, --benchmark
          Run an offline benchmark sweep and print a detailed report. Implied by the other --benchmark-* options
          
          [env: NICE_BENCHMARK=]

      --benchmark-secs <BENCHMARK_SECS>
          Approximate time budget for the benchmark sweep, in seconds
          
          [env: NICE_BENCHMARK_SECS=]
          [default: 10]

      --benchmark-upload
          Upload benchmark results without prompting
          
          [env: NICE_BENCHMARK_UPLOAD=]

      --benchmark-json
          Print the benchmark report as machine-readable JSON instead of the table; everything else (progress, upload chatter) moves to stderr so stdout is exactly one JSON document
          
          [env: NICE_BENCHMARK_JSON=]

      --telemetry
          Attach hardware/config telemetry to each submission
          
          [env: NICE_TELEMETRY=]

      --validate
          Validate results against the server before submitting
          
          [env: NICE_VALIDATE=]

      --gpu
          Use GPU acceleration (requires a build with the gpu feature). Implied by the other --gpu-* options
          
          [env: NICE_GPU=]

      --gpu-device <GPU_DEVICE>
          GPU device to use (0 for first GPU, 1 for second, etc.)
          
          [env: NICE_GPU_DEVICE=]
          [default: 0]

      --gpu-backend <GPU_BACKEND>
          Which GPU backend to use with --gpu

          Possible values:
          - auto:        Fastest measured order for the mode: detailed tries `cubecl-cuda`, `cubecl`, CUDA, then Vulkan; niceonly tries CUDA, `cubecl`, then Vulkan. See `init_gpu` for the numbers behind the ordering
          - cuda:        NVIDIA only; requires the CUDA toolkit at runtime for NVRTC
          - vulkan:      Any Vulkan 1.2 device with `shaderInt64` (AMD, Intel, NVIDIA, llvmpipe). Experimental: only present in builds with the `vulkan` feature, which the `gpu` umbrella no longer includes
          - cubecl:      `CubeCL` over wgpu: kernels written in Rust, JIT-specialized per base
          - cubecl-cuda: `CubeCL` over its native CUDA runtime (needs the `cubecl-cuda` feature and, like `cuda`, the CUDA toolkit at runtime for NVRTC)
          
          [env: NICE_GPU_BACKEND=]
          [default: auto]

      --gpu-wgpu-device <GPU_WGPU_DEVICE>
          Which wgpu adapter the `CubeCL` backend uses, in `CubeCL`'s device spelling: `DiscreteGpu(0)`, `IntegratedGpu(1)`, `Cpu`, ... Unset picks the best adapter. This exists because --gpu-device indexes a per-backend namespace (CUDA ordinals != Vulkan ordinals != wgpu adapters), so on a multi-GPU box no single number is right for every backend; the chosen adapter and its graphics API are always logged
          
          [env: NICE_GPU_WGPU_DEVICE=]

  -l, --log-level <LOG_LEVEL>
          Set the log level (overrides `RUST_LOG` environment variable)
          
          [env: NICE_LOG_LEVEL=]
          [possible values: off, error, warn, info, debug, trace]

  -h, --help
          Print help (see a summary with '-h')

  -V, --version
          Print version
```

## Project Architecture

This repository has a common library with most actual functionality included. There are two main binaries: the API server and the client. These can be run directly from source with `cargo run -p nice_api` or `cargo run -p nice_client`. There are also binaries for a deamon and some scheduled jobs, and a library for a wasm client.

There are some feature flags that enable specific dependencies:

- `nice_common/database` is set automatically from binaries that connect directly to postgres (`api` and `jobs`). This requires the `libpq-dev` package to be installed.
- `nice_client/rustls-tls` is enabled by default and uses rustls for TLS connections, which doesn't require any external dependencies. Disable it and enable `nice_client/openssl-tls` to use `openssl`.
- In order to build the client with GPU acceleration, enable the `nice_client/gpu` feature. It is an umbrella over multiple backends below:
  - `nice_client/cuda` is the hand-written CUDA backend (NVIDIA only). It requires the CUDA toolkit at runtime for NVRTC kernel compilation.
  - `nice_client/vulkan` is the hand-written WGSL backend (**experimental — not part of the `gpu` umbrella**); it runs on any Vulkan 1.2 device with `shaderInt64` (AMD, Intel, NVIDIA, llvmpipe, and MoltenVK on macOS). Every platform it serves is also covered by `cubecl`, which beats it in detailed mode on all vendors measured, so standard builds omit it; build with `--features gpu,vulkan` to include it.
  - `nice_client/cubecl` is the [CubeCL](https://github.com/tracel-ai/cubecl) backend (kernels written in Rust), running over wgpu: Vulkan on Linux/Windows, Metal on macOS. `nice_client/cubecl-cuda` adds its native CUDA runtime. Both modes run on the GPU.
  - `nice_client/cubecl-spirv` and `nice_client/cubecl-metal` (experimental, opt-in) switch the `cubecl` backend's shader compiler from naga (WGSL) to CubeCL's own SPIR-V or MSL codegen on Vulkan or Metal respectively. `cubecl-spirv` pulls in `ash`; neither is part of the `gpu` umbrella. Results are identical (device parity tests cover both); `NICE_CUBECL_WIDE=1` additionally opts into the 64-bit chunk scan on devices that expose `u64`, which is slower on every wgpu device measured so far.

Building the WASM client requires [wasm-pack](https://drager.github.io/wasm-pack/).

There are also a few scripts, to be used with [rust-script](https://rust-script.org/). You can install it with `cargo install rust-script` then run the scripts directly. It will take a while to build the first time you run it.

If you want to run a copy of this server yourself, a SQL schema file has been provided. You can build the bases and fields with the `insert_fields` script.

## GPU backends

The GPU-enabled client (`--features gpu`, or the `-gpu` docker tag) carries multiple backends in one binary and picks one at runtime; the CPU path is always available as a fallback and for verification. Kernels are JIT-compiled per base at first use, so the first field on a new base takes a few extra seconds.

| backend | runs on | needs at runtime |
|---|---|---|
| `cubecl` | any GPU via wgpu — Vulkan on Linux/Windows, Metal on macOS | a graphics driver |
| `cubecl-cuda` | NVIDIA | CUDA toolkit (NVRTC) |
| `cuda` | NVIDIA | CUDA toolkit (NVRTC) |
| `vulkan` (experimental, opt-in build) | any Vulkan 1.2 device with `shaderInt64` | a Vulkan driver; on macOS also MoltenVK + the Vulkan loader (`brew install molten-vk vulkan-loader`) |

## Why are you writing this from scratch for like the tenth time

It's the sixth time. And no comment.
