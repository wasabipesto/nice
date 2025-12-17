FROM rust:1 AS builder
WORKDIR /usr/src/nice

COPY . .
RUN cargo build -r -p nice_client

FROM debian:trixie-slim
COPY --from=builder /usr/src/nice/target/release/nice_client ./nice_client
ENTRYPOINT ["./nice_client"]
