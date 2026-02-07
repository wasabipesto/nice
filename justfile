# Requires rust:
#   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
# Requires `just` and `toml-cli` to run these helpers:
#   cargo install just toml-cli

set dotenv-load := true

# List commands, default
default:
    just --list

# Update from git and rebuild if necessary
update:
    git pull
    just build

# Tag and push a new release
tag-release version:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Checking version {{ version }}..."

    # Check that everything is committed and pushed
    if [ -n "$(git status --porcelain)" ]; then
        echo "Error: Working directory is not clean. Commit or stash changes first."
        exit 1
    fi
    if [ -n "$(git log @{u}.. 2>/dev/null)" ]; then
        echo "Error: Local commits have not been pushed to remote."
        exit 1
    fi
    echo "✓ Working directory is clean and pushed"

    # Check that tag does not already exist
    if git rev-parse "v{{ version }}" >/dev/null 2>&1; then
        echo "Error: Tag v{{ version }} already exists"
        exit 1
    fi
    echo "✓ Tag v{{ version }} does not yet exist"

    # Check that version matches Cargo.toml
    cargo_version=$(toml get Cargo.toml workspace.package.version --raw)
    if [ "$cargo_version" != "{{ version }}" ]; then
        echo "Error: Version {{ version }} does not match Cargo.toml version $cargo_version"
        exit 1
    fi
    echo "✓ Version matches Cargo.toml"

    # Check that version is in CHANGELOG.md
    if ! grep -q "## Nice v{{ version }}" CHANGELOG.md; then
        echo "Error: Version {{ version }} not found in CHANGELOG.md"
        exit 1
    fi
    echo "✓ Version found in CHANGELOG.md"

    # Create and push tag
    git tag -a v{{ version }} -m "release new version"
    git push origin v{{ version }}
    echo "✓ Tagged and pushed v{{ version }}"

# Build all packages (except wasm)
build:
    cargo build -p "*"
    cargo build -p "*" -r

# Build all packages, run all tests, and then run the client
test:
    cargo clippy -p "*"
    cargo build -p "*"
    cargo build -p "*" --features nice_client/gpu
    # cargo build -p "*" -r
    # just wasm-build
    RUST_LOG="trace" cargo test -p "*" --no-fail-fast
    # just benchmark default
    just client --validate

# List all available major dependency upgrades
cargo-upgrades:
    cargo install cargo-upgrades
    cargo upgrades

# Run client with given options
client *args:
    cargo run -r --bin nice_client -- {{ args }}

# Run benchmark
benchmark size='large':
    just client --benchmark {{ size }}

# Run the daemon
daemon *args:
    cargo run -r -p nice_daemon -- {{ args }}

# Run API server
server:
    ROCKET_ADDRESS="0.0.0.0" cargo run -r -p nice_api

# Run API server (alias)
api: server

# Run a SQL file on the database
run-sql file:
    docker exec -i nice-postgres psql \
    --username=nice \
    --dbname=nice \
    < {{ file }}

# Run scheduled jobs
jobs:
    cargo run -r -p nice_jobs

# Deploy the website and bundled assets
deploy-site:
    rclone sync web $RCLONE_SITE_TARGET --progress

# Start dev server for website
[working-directory('web')]
dev:
    python3 -m http.server

# Build WASM app and copy result to web dir
[working-directory('wasm-client')]
wasm-build:
    cargo install wasm-pack
    wasm-pack build --target web --out-dir pkg
    cp -rv pkg {{ justfile_directory() }}/web/search/

# Build WASM app and start dev server
wasm-dev: wasm-build dev

# Profile using samply, requires `cargo install samply`
profile *args:
    cargo build --profile profiling --bin nice_client
    samply record cargo run --profile profiling --bin nice_client -- --benchmark large {{ args }}
