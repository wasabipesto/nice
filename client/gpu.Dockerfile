# Runtime-only image for a prebuilt `nice_client` binary with GPU support.
# CI should place the architecture-specific binary at the root of the build context
# with the filename `nice_client`.
#
# This image includes the CUDA 12.8 runtime libraries (NVRTC for the CUDA
# backends). The base must carry a glibc at least as new as the runner that
# built the binary: CI builds on ubuntu-latest (24.04, glibc 2.39), and
# since #117 the rustls crypto backend (aws-lc-sys) references GLIBC_2.38
# symbols, so a 22.04 base (2.35) fails at load time with
# "version `GLIBC_2.38' not found". The CPU image (debian trixie, 2.41)
# was never affected. The workflow runs the binary inside the built image
# to catch this class of mismatch before pushing.

FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04 AS runtime

# Install runtime dependencies: TLS certs for HTTPS, and the CUDA runtime
# headers. The `cubecl-cuda` backend compiles its kernels with NVRTC from
# source that begins `#include <cuda_runtime.h>`, and NVRTC resolves that
# against /usr/local/cuda/include — which the `-runtime` base image does not
# ship (only the `-devel` one does). Without it every cubecl-cuda kernel fails
# to compile ("cannot open source file cuda_runtime.h") and `--gpu-backend
# cubecl-cuda` exits at init; the hand-written CUDA backend is unaffected
# because its kernel source is self-contained. cuda-cudart-dev (plus its
# cuda-cccl dependency) is ~20 MB; the version must track the base image.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    cuda-cudart-dev-12-8 \
    && rm -rf /var/lib/apt/lists/*

# Copy the prebuilt binary from the build context.
# The build context should include: ./nice_client
COPY nice_client /usr/local/bin/nice_client

# OCI metadata
LABEL org.opencontainers.image.title="Nice Client (GPU)"
LABEL org.opencontainers.image.description="a client for distributed search of square-cube pandigitals with CUDA GPU support"
LABEL org.opencontainers.image.source="https://github.com/wasabipesto/nice"

ENTRYPOINT ["/usr/local/bin/nice_client"]
