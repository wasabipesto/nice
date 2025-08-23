set dotenv-load

# List commands, default
default:
  just --list

# Run all tests
test:
    cargo test

# Run client continuously
client mode='detailed':
    cargo run -r --bin nice_client -- --repeat {{mode}}

# Run benchmark
benchmark size='large':
    cargo run -r --bin nice_client -- --quiet --benchmark {{size}}

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
build-wasm:
    wasm-pack build --target web --out-dir pkg
    cp -rv pkg {{justfile_dir()}}/web/search/

# Build WASM app and start dev server
wasm: build-wasm dev
