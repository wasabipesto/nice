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

# Request 5 fields at once (reduces connection overhead)
./nice_client --batch-size 5

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

  -u, --username <USERNAME>
          The username to send alongside your contribution

          [env: NICE_USERNAME=]
          [default: anonymous]

  -r, --repeat
          Run indefinitely with the current settings

          [env: NICE_REPEAT=]

  -q, --quiet
          Suppress all output

          [env: NICE_QUIET=]

  -v, --verbose
          Show additional output

          [env: NICE_VERBOSE=]

  -t, --threads <THREADS>
          Run parallel with this many threads

          [env: NICE_THREADS=]
          [default: 4]

  -b, --benchmark <BENCHMARK>
          Run an offline benchmark

          Possible values:
          - default:     The default benchmark range: 1e6 @ base 40
          - large:       A large benchmark range: 1e8 @ base 40
          - extra-large: A very large benchmark range: 1e9 @ base 40. This is the size of a typical field from the server
          - hi-base:     A benchmark range at a higher range: 1e6 @ base 80

          [env: NICE_BENCHMARK=]

      --validate
          Validate results against the server before submitting

          [env: NICE_VALIDATE=]

      --gpu
          Use GPU acceleration (requires gpu feature)

          [env: NICE_GPU=]

      --gpu-device <GPU_DEVICE>
          CUDA device to use for GPU processing (0 for first GPU, 1 for second, etc.)

          [env: NICE_GPU_DEVICE=]
          [default: 0]

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
- In order to build the client with GPU acceleration, enable the `nice_client/gpu` feature. This requires no additional build-time dependencies, but it does require the CUDA toolkit to be available at runtime for kernel compilation.

Building the WASM client requires [wasm-pack](https://drager.github.io/wasm-pack/).

There are also a few scripts, to be used with [rust-script](https://rust-script.org/). You can install it with `cargo install rust-script` then run the scripts directly. It will take a while to build the first time you run it.

If you want to run a copy of this server yourself, a SQL schema file has been provided. You can build the bases and fields with the `insert_fields` script.

## Why are you writing this from scratch for like the tenth time

It's the sixth time. And no comment.
