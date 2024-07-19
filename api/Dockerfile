FROM rust:1.79 as builder
WORKDIR /usr/src/nice

COPY . .
RUN cargo test
RUN cargo build -r -p nice_api

FROM debian:bookworm-slim
WORKDIR /usr/src/nice

RUN apt-get update
RUN apt-get install libpq5 -y
COPY --from=builder /usr/src/nice/target/release/nice_api .

CMD ["./nice_api"]