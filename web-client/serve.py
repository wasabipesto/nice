#!/usr/bin/env python3
"""
Simple HTTP server for serving the Nice Numbers Web Client.
This server adds the necessary CORS headers and serves the WASM files correctly.
"""

import http.server
import socketserver
import os
import sys
from pathlib import Path


class CORSHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP request handler with CORS headers for WASM support"""

    def end_headers(self):
        # Add CORS headers
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

        # Set correct MIME types for WASM files
        if self.path.endswith(".wasm"):
            self.send_header("Content-Type", "application/wasm")
        elif self.path.endswith(".js"):
            self.send_header("Content-Type", "application/javascript")

        super().end_headers()

    def do_OPTIONS(self):
        """Handle preflight OPTIONS requests"""
        self.send_response(200)
        self.end_headers()


def main():
    # Default port
    port = 8000

    # Parse command line arguments
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print(f"Invalid port number: {sys.argv[1]}")
            sys.exit(1)

    # Check if we're in the right directory
    if not Path("index.html").exists():
        print("Error: index.html not found in current directory.")
        print("Make sure you're running this from the web-client directory.")
        sys.exit(1)

    # Check if WASM files exist
    pkg_dir = Path("pkg")
    if not pkg_dir.exists():
        print("Warning: pkg/ directory not found.")
        print("Run './build.sh' or 'wasm-pack build --target web --out-dir pkg' first.")
        print("Serving anyway in case you want to test the build process...")

    # Start server
    try:
        with socketserver.TCPServer(("", port), CORSHTTPRequestHandler) as httpd:
            print(f"🚀 Nice Numbers Web Client Server")
            print(f"📁 Serving directory: {os.getcwd()}")
            print(f"🌐 Server running at: http://localhost:{port}")
            print(f"📋 Open your browser and navigate to the URL above")
            print(f"🛑 Press Ctrl+C to stop the server")
            print()

            if pkg_dir.exists():
                wasm_files = list(pkg_dir.glob("*.wasm"))
                if wasm_files:
                    print(
                        f"✅ Found WASM files: {', '.join(f.name for f in wasm_files)}"
                    )
                else:
                    print("⚠️  No WASM files found in pkg/ directory")

            print()
            httpd.serve_forever()

    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except OSError as e:
        if e.errno == 98:  # Address already in use
            print(f"❌ Port {port} is already in use.")
            print(f"💡 Try a different port: python3 serve.py {port + 1}")
        else:
            print(f"❌ Error starting server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
