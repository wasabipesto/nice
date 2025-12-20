# Runtime-only image for a prebuilt `nice_client` binary.
# CI should place the architecture-specific binary at the root of the build context
# with the filename `nice_client` (same name for amd64 and arm64 builds).

FROM debian:trixie-slim AS runtime

# Install runtime dependencies (TLS certs for HTTPS, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy the prebuilt binary from the build context.
# The build context should include: ./nice_client
COPY nice_client /usr/local/bin/nice_client

# OCI metadata
LABEL org.opencontainers.image.title="Nice Client"
LABEL org.opencontainers.image.description="a client for distributed search of square-cube pandigitals"
LABEL org.opencontainers.image.source="https://github.com/wasabipesto/nice"

ENTRYPOINT ["/usr/local/bin/nice_client"]
