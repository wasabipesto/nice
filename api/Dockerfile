FROM rust:1 AS chef
RUN cargo install cargo-chef
WORKDIR /usr/src/nice

FROM chef AS planner
COPY . .
RUN cargo chef prepare --recipe-path recipe.json

FROM chef AS builder
COPY --from=planner /usr/src/nice/recipe.json recipe.json
# Build dependencies - this is the caching Docker layer!
RUN cargo chef cook --release --recipe-path recipe.json

# Build application
COPY . .
RUN cargo test
# The commit this image was built from, surfaced by /status. `.dockerignore`
# keeps `.git` out of the build context, so it has to come in as an argument:
#   docker build --build-arg NICE_BUILD_COMMIT=$(git rev-parse --short HEAD) ...
# Builds that omit it report a null commit.
ARG NICE_BUILD_COMMIT
RUN cargo build -r -p nice_api

FROM debian:bookworm-slim
WORKDIR /usr/src/nice

RUN apt-get update && apt-get install libpq5 -y && rm -rf /var/lib/apt/lists/*
COPY --from=builder /usr/src/nice/target/release/nice_api .

CMD ["./nice_api"]
