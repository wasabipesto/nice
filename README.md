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
          Run an offline benchmark sweep and print a detailed report
          
          [env: NICE_BENCHMARK=]

      --benchmark-secs <BENCHMARK_SECS>
          Approximate time budget for the benchmark sweep, in seconds
          
          [env: NICE_BENCHMARK_SECS=]
          [default: 10]

      --benchmark-upload
          Upload benchmark results without prompting
          
          [env: NICE_BENCHMARK_UPLOAD=]

      --telemetry
          Attach hardware/config telemetry to each submission
          
          [env: NICE_TELEMETRY=]

      --validate
          Validate results against the server before submitting
          
          [env: NICE_VALIDATE=]

      --gpu
          Use GPU acceleration (requires gpu feature)
          
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
          - cubecl-cuda: `CubeCL` over its native CUDA runtime (needs the `cubecl-cuda` feature)
          
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

## GPU acceleration

The GPU-enabled client (`--features gpu`, or the `-gpu` docker tag) carries
four backends in one binary and picks one at runtime; the CPU path is always
available as a fallback and for verification. Kernels are JIT-compiled per
base at first use, so the first field on a new base takes a few extra seconds.

| backend | runs on | needs at runtime |
|---|---|---|
| `cubecl` | any GPU via wgpu — Vulkan on Linux/Windows, Metal on macOS | a graphics driver; nothing else |
| `cubecl-cuda` | NVIDIA | CUDA toolkit (NVRTC) |
| `cuda` | NVIDIA | CUDA toolkit (NVRTC) |
| `vulkan` (experimental, opt-in build) | any Vulkan 1.2 device with `shaderInt64` | a Vulkan driver; on macOS also MoltenVK + the Vulkan loader (`brew install molten-vk vulkan-loader`) |

Nothing needs GPU libraries at *build* time — every backend loads its driver
dynamically, so one binary built anywhere runs anywhere.

**Backend selection.** `--gpu-backend auto` (the default) tries backends in
the measured-fastest order for the mode: detailed tries `cubecl-cuda`,
`cubecl`, then `cuda`; niceonly tries `cuda`, then `cubecl` (each list ends
with `vulkan` in the opt-in builds that include it). A backend that fails to
initialize falls through to the next; a
backend you name explicitly is fatal if it fails, so a distributed client
never silently drops to a slower path. The init log always prints which
backend and device won, and benchmark/telemetry reports carry both.

**Multi-GPU machines.** `--gpu-device N` selects the device for the CUDA and
Vulkan backends — but each backend numbers devices in its own order, so on
mixed-GPU machines (e.g. a laptop with an iGPU) the right N can differ per
backend; check the init log. The wgpu-based `cubecl` backend uses a typed
selector instead: `--gpu-wgpu-device "DiscreteGpu(0)"` (or
`IntegratedGpu(1)`, `Cpu`, ...), since wgpu enumerates adapters by kind.
Unset, it picks the highest-powered adapter.

**Troubleshooting.**

- *"CubeCL CUDA smoke kernel did not run (is the CUDA toolkit, including
  NVRTC, installed?)"* — you have an NVIDIA driver but not the CUDA toolkit.
  Install the toolkit, or let `auto` fall through to `cubecl`, which needs
  only the driver.
- *"GPU histogram counted 0 of N candidates"* — the device dropped work. On
  wgpu this usually means the driver's watchdog reset the GPU (check `dmesg`
  for `ring gfx ... timeout`); please report it, since batches are sized to
  stay under watchdogs.
- *Browser client on Linux* — WebGPU is still rolling out there, so which
  browser you use decides whether the GPU option appears at all. Measured on
  one AMD RX 9070 XT (RADV, Wayland):
  - **Firefox Nightly works**, and is the recommended way to use the GPU
    path on Linux today — a full field runs on the GPU with no flags.
  - **Firefox release (152+)** has WebGPU behind `dom.webgpu.enabled` in
    `about:config`. If the page still reports no adapter, also try
    `dom.webgpu.workers.enabled` (this client runs WebGPU inside a Worker)
    and `gfx.webgpu.ignore-blocklist` (Firefox ships a conservative adapter
    blocklist on Linux, and a blocklisted adapter is reported as no adapter).
  - **LibreWolf 149 does not work**, even with those prefs set: it caps GPU
    memory at roughly 10 MB in total, refuses single allocations of 8 MB and
    up, and does not appear to reclaim on release. Stock Firefox on the same
    machine has no such ceiling, so this is LibreWolf's hardening (it ships
    with `resistFingerprinting` on and WebGL disabled), not a Firefox
    limitation.
  - **Chromium offers no adapter** on AMD yet — Chrome's Linux rollout is
    per-vendor and AMD is not in it.

  In every failing case the page falls back to the CPU worker pool
  automatically; nothing needs to be configured for that.
- *"Unable to find a Vulkan driver"* on macOS — the experimental `vulkan` backend needs
  `brew install molten-vk vulkan-loader` and
  `VK_ICD_FILENAMES=/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json`. The
  `cubecl` backend needs none of that (it talks to Metal directly) and is
  faster in detailed mode.
- *Wrong GPU on a laptop* — the init log names the chosen adapter (and its
  graphics API, since wgpu may pick DX12 on Windows); pin it with
  `--gpu-device` / `--gpu-wgpu-device` as above.
- *A base falls back to the CPU* with a warning — bases outside the GPU range
  (above ~b97) process on the CPU by design; results are identical.

## Project Architecture

This repository has a common library with most actual functionality included. There are two main binaries: the API server and the client. These can be run directly from source with `cargo run -p nice_api` or `cargo run -p nice_client`. There are also binaries for a deamon and some scheduled jobs, and a library for a wasm client.

There are some feature flags that enable specific dependencies:

- `nice_common/database` is set automatically from binaries that connect directly to postgres (`api` and `jobs`). This requires the `libpq-dev` package to be installed.
- `nice_client/rustls-tls` is enabled by default and uses rustls for TLS connections, which doesn't require any external dependencies. Disable it and enable `nice_client/openssl-tls` to use `openssl`.
- In order to build the client with GPU acceleration, enable the `nice_client/gpu` feature. It is an umbrella over every backend below, so the release recipe is two builds: default features for a lightweight CPU-only binary, `--features gpu` for one binary that runs on the CPU or any supported GPU. No backend needs its GPU libraries at *build* time — each one `dlopen`s its driver at runtime — and the backend is selected at runtime with `--gpu-backend` (`auto` picks the measured-fastest order per mode: detailed tries `cubecl-cuda`, `cubecl`, `cuda`, then `vulkan`; niceonly tries `cuda`, `cubecl`, then `vulkan`).
- `nice_client/cuda` is the hand-written CUDA backend (NVIDIA only). It requires the CUDA toolkit at runtime for NVRTC kernel compilation.
- `nice_client/vulkan` is the hand-written WGSL backend (**experimental — not part of the `gpu` umbrella**); it runs on any Vulkan 1.2 device with `shaderInt64` (AMD, Intel, NVIDIA, llvmpipe, and MoltenVK on macOS). Every platform it serves is also covered by `cubecl`, which beats it in detailed mode on all vendors measured, so standard builds omit it; build with `--features gpu,vulkan` to include it while its fate (promote or remove) is decided.
- `nice_client/cubecl` is the [CubeCL](https://github.com/tracel-ai/cubecl) backend (kernels written in Rust), running over wgpu: Vulkan on Linux/Windows, Metal on macOS. `nice_client/cubecl-cuda` adds its native CUDA runtime. Both modes run on the GPU.

Building the WASM client requires [wasm-pack](https://drager.github.io/wasm-pack/).

There are also a few scripts, to be used with [rust-script](https://rust-script.org/). You can install it with `cargo install rust-script` then run the scripts directly. It will take a while to build the first time you run it.

If you want to run a copy of this server yourself, a SQL schema file has been provided. You can build the bases and fields with the `insert_fields` script.

### GPU development

The CUDA kernels in `common/src/cuda` are compiled at runtime by NVRTC, once
per (base, mode), with all base-dependent constants baked in as defines. Most
of the kernel logic is covered by CPU-side mirror tests that run with the
normal test suite, but the kernel source itself is only parsed by NVRTC — and
NVRTC is just a shared library, so you can compile-test every base's kernel
on a machine with no GPU at all. The `nvrtc_compiles_kernels_for_all_supported_bases`
test activates automatically when `libnvrtc` is loadable and skips otherwise.

The easiest way to get `libnvrtc` is the pip wheel, via [uv](https://docs.astral.sh/uv/):

```sh
# Fetch libnvrtc from the pip wheel and point the linker at it (no GPU needed)
NVRTC_LIB=$(uv run --no-project --with nvidia-cuda-nvrtc-cu12 python -c \
    "import nvidia.cuda_nvrtc; print(nvidia.cuda_nvrtc.__path__[0] + '/lib')")
LD_LIBRARY_PATH="$NVRTC_LIB" cargo test --features cuda -p nice_common
```

Or run the tests inside a CUDA container, which ships `libnvrtc`:

```sh
docker run --rm -v "$PWD":/work -w /work nvidia/cuda:12.4.1-devel-ubuntu22.04 \
    bash -c "apt-get update -qq && apt-get install -y -qq curl build-essential > /dev/null && \
             curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y -q && \
             . ~/.cargo/env && cargo test --features cuda -p nice_common"
```

Functional GPU tests (CPU/GPU result parity) still need real hardware; they
are `#[ignore]`d and run with `cargo test --features cuda -- --ignored`.

## Why are you writing this from scratch for like the tenth time

It's the sixth time. And no comment.
