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

# Build application. .git is not in the docker context, so the build commit
# has to come in as an argument: `docker build --build-arg NICE_BUILD_SHA=$(git rev-parse HEAD) ...`
ARG NICE_BUILD_SHA=unknown
ENV NICE_BUILD_SHA=$NICE_BUILD_SHA
COPY . .
RUN cargo test
RUN cargo build -r -p nice_api

FROM debian:bookworm-slim
WORKDIR /usr/src/nice

RUN apt-get update && apt-get install libpq5 -y && rm -rf /var/lib/apt/lists/*
COPY --from=builder /usr/src/nice/target/release/nice_api .

CMD ["./nice_api"]
