FROM rust:1 AS builder
WORKDIR /usr/src/nice

# Copy workspace files
COPY . .

# Build the application
RUN cargo build -r -p nice_client

# Ship a slim runner image
FROM debian:trixie-slim

# Copy the binary
COPY --from=builder /usr/src/nice/target/release/nice_client /usr/local/bin/nice_client

# Add metadata
LABEL org.opencontainers.image.title="Nice Client"
LABEL org.opencontainers.image.description="a client for distributed search of square-cube pandigitals"
LABEL org.opencontainers.image.source="https://github.com/wasabipesto/nice"

ENTRYPOINT ["nice_client"]
