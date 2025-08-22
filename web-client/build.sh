#!/bin/bash

# Build script for Nice Numbers WebAssembly client
set -e

echo "Building Nice Numbers WebAssembly client..."

# Check if wasm-pack is installed
if ! command -v wasm-pack &> /dev/null; then
    echo "Error: wasm-pack is not installed."
    echo "Install it with: curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh"
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "Cargo.toml" ]; then
    echo "Error: Cargo.toml not found. Make sure you're in the web-client directory."
    exit 1
fi

# Build the WASM package
echo "Compiling Rust to WebAssembly..."
wasm-pack build --target web --out-dir pkg

# Check if build was successful
if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
    echo ""
    echo "Generated files:"
    ls -la pkg/
    echo ""
    echo "To run the client:"
    echo "1. Serve this directory with an HTTP server (required for WASM modules)"
    echo "2. Open index.html in your browser"
    echo ""
    echo "Example server commands:"
    echo "  Python 3: python3 -m http.server 8000"
    echo "  Python 2: python -m SimpleHTTPServer 8000"
    echo "  Node.js: npx serve ."
    echo "  PHP: php -S localhost:8000"
    echo ""
    echo "Then visit: http://localhost:8000"
else
    echo "❌ Build failed!"
    exit 1
fi
