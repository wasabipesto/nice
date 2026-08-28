#!/bin/bash
# Run the browser E2E in the playwright container (no host browser needed).
# SwiftShader supplies a software WebGPU adapter, so the GPU path is tested
# end to end without GPU hardware. Usage: ./run.sh [--skip-gpu]
set -euo pipefail
cd "$(dirname "$0")"
REPO_ROOT="$(cd ../.. && pwd)"
IMG=mcr.microsoft.com/playwright:v1.62.0-noble
docker run --rm -v "$REPO_ROOT":/repo -w /repo/wasm-client/e2e \
    --ipc=host "$IMG" bash -c "
        npm i --no-save --no-audit --no-fund playwright@1.62.0 >/dev/null 2>&1 &&
        node e2e.mjs $*"
