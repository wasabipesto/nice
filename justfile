set dotenv-load

version := shell("toml get common/Cargo.toml package.version")

# List commands, default
default:
  just --list

# Build all packages and run all tests
test:
    cargo build
    cargo build -r
    cargo test
    cargo clippy

# List all available dependency upgrades
upgrades:
    cargo upgrades --manifest-path api/Cargo.toml
    cargo upgrades --manifest-path client/Cargo.toml
    cargo upgrades --manifest-path common/Cargo.toml
    cargo upgrades --manifest-path jobs/Cargo.toml
    cargo upgrades --manifest-path wasm-client/Cargo.toml

# Build and run a Dockerfile (for building against a specific glibc)
docker dockerfile:
    docker build -t nice-{{file_stem(dockerfile)}} -f {{dockerfile}} .
    docker run -it -v .:/opt/nice nice-{{file_stem(dockerfile)}}

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
wasm-setup:
    wasm-pack build --target web --out-dir pkg
    cp -rv pkg {{justfile_dir()}}/web/search/

# Build WASM app and start dev server
wasm-dev: wasm-setup dev
