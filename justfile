# Requires rust:
#   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
# Requires `just` and `toml-cli` to run these helpers:
#   cargo install just toml-cli

set dotenv-load

version := shell("toml get common/Cargo.toml package.version")

# List commands, default
default:
  just --list

# Update from git and rebuild if necessary
update:
    git pull
    just build

# Build all packages (except wasm)
build:
    cargo build
    cargo build -r

# Build all packages, run all tests, and then run the client
test:
    cargo clippy
    cargo build
    cargo build -r
    just wasm-build
    cargo test --no-fail-fast
    just benchmark default
    just client

# List all available major dependency upgrades
cargo-upgrades:
    cargo install cargo-upgrades
    cargo upgrades

# Build and run a Dockerfile (for building against a specific glibc)
docker dockerfile:
    docker build -t nice-{{lowercase(file_stem(dockerfile))}} -f {{dockerfile}} .
    docker run -it -v .:/opt/nice nice-{{lowercase(file_stem(dockerfile))}}

# Build each version in docker and copy artifacts for release
build-for-release:
    mkdir -p release
    just docker docker/bullseye.Dockerfile
    cp target-bullseye/release/nice_client release/nice-client-{{version}}-x86_64-bullseye
    just docker docker/bookworm.Dockerfile
    cp target-bookworm/release/nice_client release/nice-client-{{version}}-x86_64-bookworm
    just docker docker/trixie.Dockerfile
    cp target-trixie/release/nice_client release/nice-client-{{version}}-x86_64-trixie

# Run client with given options
client *args:
    cargo run -r --bin nice_client -- {{args}}

# Run benchmark
benchmark size='large':
    just client --benchmark {{size}}

# Run the daemon
daemon *args:
    cargo run -r --bin nice_daemon -- {{args}}

# Run API server
server:
    cargo run -r --bin nice_api

# Run scheduled jobs
jobs:
    cargo run -r --bin nice_jobs

# Deploy the website and bundled assets
deploy-site:
    rclone sync web $RCLONE_SITE_TARGET --progress

# Start dev server for website
[working-directory: 'web']
dev:
    python3 -m http.server

# Build WASM app and copy result to web dir
[working-directory: 'wasm-client']
wasm-build:
    cargo install wasm-pack
    wasm-pack build --target web --out-dir pkg
    cp -rv pkg {{justfile_dir()}}/web/search/

# Build WASM app and start dev server
wasm-dev: wasm-build dev
