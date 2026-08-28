#!/usr/bin/env bash
# Download third-party site assets that aren't committed to the repo.
# Called by `just vendor-fetch` (and the e2e harness, which serves the
# search page standalone and needs its copy of Plot in place).
#
# Plot's UMD build and the d3 global it expects. The CDN's ESM build fans
# out into 48 requests across six dependency levels, all of which a page
# has to wait on before it can fetch any data.
set -euo pipefail
cd "$(dirname "$0")/.."

fetch() {
    local dir="$1"
    mkdir -p "$dir"
    curl -sSfL "https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js" \
        -o "$dir/d3.min.js"
    curl -sSfL "https://cdn.jsdelivr.net/npm/@observablehq/plot@0.6.17/dist/plot.umd.min.js" \
        -o "$dir/plot.umd.min.js"
    sha256sum -c - <<CHECKSUMS
f2094bbf6141b359722c4fe454eb6c4b0f0e42cc10cc7af921fc158fceb86539  $dir/d3.min.js
4358086467740777dd788d6b27a95cebdbaefdd50c730a3060117073bd7134cb  $dir/plot.umd.min.js
CHECKSUMS
}

# The index page and the search page each serve from their own directory;
# the search page keeps a local copy so the e2e harness (which serves
# web/search as its root) and hardware test copies stay self-contained.
fetch web/vendor
mkdir -p web/search/vendor
cp web/vendor/d3.min.js web/vendor/plot.umd.min.js web/search/vendor/
