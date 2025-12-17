# Use cargo-chef for dependency caching
FROM lukemathwalker/cargo-chef:latest-rust-1 AS chef
WORKDIR /usr/src/nice

# Prepare the recipe file
FROM chef AS planner
COPY . .
RUN cargo chef prepare --recipe-path recipe.json

# Build dependencies - this layer will be cached unless dependencies change
FROM chef AS builder
COPY --from=planner /usr/src/nice/recipe.json recipe.json
RUN cargo chef cook --release --locked --recipe-path recipe.json -p nice_client

# Build the application
COPY . .
RUN cargo build --release --locked -p nice_client

# Ship a slim runner image
FROM debian:trixie-slim AS runtime

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy the binary
COPY --from=builder /usr/src/nice/target/release/nice_client /usr/local/bin/nice_client

# Add metadata
LABEL org.opencontainers.image.title="Nice Client"
LABEL org.opencontainers.image.description="a client for distributed search of square-cube pandigitals"
LABEL org.opencontainers.image.source="https://github.com/wasabipesto/nice"

ENTRYPOINT ["nice_client"]
