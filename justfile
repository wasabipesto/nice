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
