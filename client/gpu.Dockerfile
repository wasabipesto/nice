# Runtime-only image for a prebuilt `nice_client` binary with GPU support.
# CI should place the architecture-specific binary at the root of the build context
# with the filename `nice_client`.
#
# This image includes CUDA 12.0 runtime libraries required for GPU acceleration.

FROM nvidia/cuda:12.0.0-runtime-ubuntu22.04 AS runtime

# Install runtime dependencies (TLS certs for HTTPS, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy the prebuilt binary from the build context.
# The build context should include: ./nice_client
COPY nice_client /usr/local/bin/nice_client

# OCI metadata
LABEL org.opencontainers.image.title="Nice Client (GPU)"
LABEL org.opencontainers.image.description="a client for distributed search of square-cube pandigitals with CUDA GPU support"
LABEL org.opencontainers.image.source="https://github.com/wasabipesto/nice"

ENTRYPOINT ["/usr/local/bin/nice_client"]
