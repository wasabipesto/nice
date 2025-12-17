# Nice!

> Join the distributed search for square-cube pandigitals!

## Why does this exist

Square-cube pandigials ("nice" numbers) seem to be distributed pseudo-randomly. It doesn't take very long to check if a number is pandigital in a specific base, but even after we narrow the search range to numbers with the right amount of digits in their square and cube there's a lot of numbers to check. This system coordinates multiple clients to search more efficiently.

For more background, check out the [original article](https://beautifulthorns.wixsite.com/home/post/is-69-unique) and [my findings](https://nicenumbers.net).

## Client Quickstart

The easiest way to get started is by going to [https://nicenumbers.net/search/](https://nicenumbers.net/search) and running it in your browser. You'll see live results and everything will be submitted in your name.

If you want to go even faster, you can run the native binaries. Currently the CI builds are disabled but you can build it fairly easily if you have the [rust toolchain](https://www.rust-lang.org/tools/install) installed.

```
# Clone the repository
git clone git@github.com:wasabipesto/nice.git
cd nice

# Build the client binary
cargo build -r -p nice_client
cd target/release

# Run once with default settings
./nice_client
```

A typical use-case involves running multiple clients continuously, in background, with a particular username. You can accomplish this with a combination of something like `screen` and the built-in repeat option:

```
screen -dm ./nice_client -r -u your_name
```

You can find various settings and their options with the `--help` flag:

```
Usage: nice_client [OPTIONS] [MODE]

Arguments:
  [MODE]
          The checkout mode to use

          [default: detailed]

          Possible values:
          - detailed: Get detailed stats on all numbers, important for long-term analytics
          - niceonly: Implements optimizations to speed up the search, usually by a factor of around 20. Does not keep statistics and cannot be quickly verified

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

  -t, --threads <THREADS>
          Run parallel with this many threads

          [default: 4]

  -b, --benchmark <BENCHMARK>
          Run an offline benchmark

          Possible values:
          - default:     The default benchmark range: 1e6 @ base 40
          - large:       A large benchmark range: 1e8 @ base 40
          - extra-large: A very large benchmark range: 1e9 @ base 40. This is the size of a typical field from the server
          - hi-base:     A benchmark range at a higher range: 1e6 @ base 80

  -h, --help
          Print help (see a summary with '-h')

  -V, --version
          Print version
```

## Project Architecture

This repository has a common library with most actual functionality included. There are two main binaries: the API server and the client. These can be run directly from source with `cargo run -p nice_api` or `cargo run -p nice_client`. There are also binaries for a deamon and some scheduled jobs, and a library for a wasm client.

There are also a few scripts, to be used with [rust-script](https://rust-script.org/). You can install it with `cargo install rust-script` then run the scripts directly. It will take a while to build the first time you run it.

If you want to run a copy of this server yourself, a SQL schema file has been provided. You can build the bases and fields with the `insert_fields` script.

## Why are you writing this from scratch for like the tenth time

It's the sixth time. And no comment.
