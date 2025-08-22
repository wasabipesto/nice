#!/bin/bash

# Nice Numbers Web Client - Web Worker Version - Setup and Run Script
# This script handles the complete setup and execution of the web worker client

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_header() {
    echo -e "\n${BLUE}🧵 Nice Numbers Web Worker Client${NC}"
    echo -e "${BLUE}=====================================${NC}\n"
}

# Check if we're in the right directory
check_directory() {
    if [ ! -f "Cargo.toml" ] || [ ! -f "index-worker.html" ]; then
        print_error "Not in the web-client directory or worker files missing!"
        print_info "Please run this script from the nice/web-client directory"
        exit 1
    fi
    print_success "In correct directory"
}

# Check if Rust is installed
check_rust() {
    if ! command -v rustc &> /dev/null; then
        print_error "Rust is not installed!"
        print_info "Install Rust from: https://rustup.rs/"
        print_info "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
        exit 1
    fi

    local rust_version=$(rustc --version)
    print_success "Rust found: $rust_version"
}

# Check if wasm-pack is installed
check_wasm_pack() {
    if ! command -v wasm-pack &> /dev/null; then
        print_error "wasm-pack is not installed!"
        print_info "Installing wasm-pack..."

        if command -v curl &> /dev/null; then
            curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
            if [ $? -eq 0 ]; then
                print_success "wasm-pack installed successfully"
            else
                print_error "Failed to install wasm-pack"
                exit 1
            fi
        else
            print_error "curl not found. Please install wasm-pack manually:"
            print_info "Visit: https://rustwasm.github.io/wasm-pack/installer/"
            exit 1
        fi
    else
        local wasm_pack_version=$(wasm-pack --version)
        print_success "wasm-pack found: $wasm_pack_version"
    fi
}

# Check if Python is available for serving
check_python() {
    if command -v python3 &> /dev/null; then
        local python_version=$(python3 --version)
        print_success "Python3 found: $python_version"
        return 0
    elif command -v python &> /dev/null; then
        local python_version=$(python --version)
        print_success "Python found: $python_version"
        return 0
    else
        print_warning "Python not found. You'll need to serve the files manually."
        return 1
    fi
}

# Check browser support for Web Workers
check_worker_files() {
    if [ ! -f "worker.js" ]; then
        print_error "worker.js not found!"
        print_info "The Web Worker script is required for this version"
        exit 1
    fi

    if [ ! -f "index-worker.html" ]; then
        print_error "index-worker.html not found!"
        print_info "The Web Worker HTML file is required"
        exit 1
    fi

    print_success "Web Worker files found"
}

# Build the WASM package
build_wasm() {
    print_info "Building WebAssembly package for Web Worker..."

    # Clean previous build
    if [ -d "pkg" ]; then
        rm -rf pkg
        print_info "Cleaned previous build"
    fi

    # Build with wasm-pack targeting web
    wasm-pack build --target web --out-dir pkg

    if [ $? -eq 0 ] && [ -d "pkg" ]; then
        print_success "WASM build completed successfully!"

        # Check if the required function is exposed
        if grep -q "get_num_unique_digits_wasm" pkg/nice_web_client.js; then
            print_success "Worker-specific functions are properly exposed"
        else
            print_warning "Worker function might not be exposed correctly"
        fi

        # List generated files
        print_info "Generated files:"
        ls -la pkg/ | grep -E '\.(js|wasm|ts)$' || true
    else
        print_error "WASM build failed!"
        exit 1
    fi
}

# Create a temporary HTML file that defaults to the worker version
create_index_redirect() {
    if [ ! -f "index.html.backup" ]; then
        if [ -f "index.html" ]; then
            cp index.html index.html.backup
            print_info "Backed up original index.html"
        fi
    fi

    # Create a simple redirect to the worker version
    cat > index.html << 'EOF'
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Nice Numbers - Redirecting to Web Worker Version</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            margin: 0;
            padding: 40px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: white;
            display: flex;
            align-items: center;
            justify-content: center;
            text-align: center;
        }
        .container {
            background: rgba(255, 255, 255, 0.1);
            padding: 40px;
            border-radius: 12px;
            backdrop-filter: blur(10px);
        }
        h1 { margin-bottom: 20px; }
        p { margin: 10px 0; }
        a {
            color: #fff;
            text-decoration: underline;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🧵 Nice Numbers Web Worker Client</h1>
        <p>Redirecting to the Web Worker version for better performance...</p>
        <p>If you're not redirected automatically, <a href="index-worker.html">click here</a></p>
        <p><small>Web Workers prevent browser freezing during computation</small></p>
    </div>
    <script>
        // Redirect after a short delay
        setTimeout(() => {
            window.location.href = 'index-worker.html';
        }, 2000);
    </script>
</body>
</html>
EOF

    print_success "Created redirect to Web Worker version"
}

# Start the development server
start_server() {
    local port=${1:-8000}

    print_info "Starting development server on port $port..."
    print_info "🧵 This will serve the Web Worker version by default"

    if [ -f "serve.py" ]; then
        print_info "Using custom Python server with WASM support..."
        python3 serve.py $port
    elif command -v python3 &> /dev/null; then
        print_info "Using Python3 built-in server..."
        python3 -m http.server $port
    elif command -v python &> /dev/null; then
        print_info "Using Python2 built-in server..."
        python -m SimpleHTTPServer $port
    else
        print_error "No suitable server found!"
        print_info "Please install Python or serve the files manually with:"
        print_info "- Any HTTP server pointing to this directory"
        print_info "- Then open: http://localhost:$port/index-worker.html"
        exit 1
    fi
}

# Restore original files
cleanup() {
    if [ -f "index.html.backup" ]; then
        mv index.html.backup index.html
        print_info "Restored original index.html"
    fi
}

# Display usage information
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "This script builds and serves the Web Worker version of the Nice Numbers client."
    echo "The Web Worker version prevents browser freezing during computation."
    echo ""
    echo "Options:"
    echo "  --build-only     Build WASM package only (don't start server)"
    echo "  --serve-only     Start server only (skip build)"
    echo "  --port PORT      Use specific port (default: 8000)"
    echo "  --no-redirect    Don't create index.html redirect"
    echo "  --cleanup        Restore original files and exit"
    echo "  --help           Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                    # Build and serve Web Worker version on port 8000"
    echo "  $0 --port 3000       # Build and serve on port 3000"
    echo "  $0 --build-only      # Just build, don't serve"
    echo "  $0 --serve-only      # Just serve (assumes already built)"
    echo "  $0 --cleanup         # Restore original index.html"
}

# Trap to cleanup on exit
trap cleanup EXIT

# Main script logic
main() {
    local build_only=false
    local serve_only=false
    local port=8000
    local no_redirect=false
    local cleanup_only=false

    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --build-only)
                build_only=true
                shift
                ;;
            --serve-only)
                serve_only=true
                shift
                ;;
            --port)
                port="$2"
                shift 2
                ;;
            --no-redirect)
                no_redirect=true
                shift
                ;;
            --cleanup)
                cleanup_only=true
                shift
                ;;
            --help)
                show_usage
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done

    # Handle cleanup-only mode
    if [ "$cleanup_only" = true ]; then
        cleanup
        print_success "Cleanup completed"
        exit 0
    fi

    # Validate port number
    if ! [[ "$port" =~ ^[0-9]+$ ]] || [ "$port" -lt 1 ] || [ "$port" -gt 65535 ]; then
        print_error "Invalid port number: $port"
        exit 1
    fi

    print_header

    # Run checks
    check_directory
    check_worker_files

    if [ "$serve_only" = false ]; then
        check_rust
        check_wasm_pack
        build_wasm
    fi

    if [ "$build_only" = false ]; then
        check_python

        # Create redirect unless disabled
        if [ "$no_redirect" = false ]; then
            create_index_redirect
        fi

        echo ""
        print_success "Setup complete! Starting Web Worker server..."
        print_info "🧵 Web Workers prevent browser freezing during computation"
        print_info "Open your browser and navigate to: http://localhost:$port"
        print_info "Direct link to worker version: http://localhost:$port/index-worker.html"
        print_warning "Press Ctrl+C to stop the server"
        echo ""

        # Give user a moment to read the instructions
        sleep 2

        start_server $port
    else
        print_success "Build completed! To serve the Web Worker files:"
        print_info "Run: $0 --serve-only --port $port"
        print_info "Or use any HTTP server to serve this directory"
        print_info "Direct URL: http://localhost:$port/index-worker.html"
    fi
}

# Handle Ctrl+C gracefully
trap 'echo -e "\n${YELLOW}🛑 Stopping Web Worker server...${NC}"; cleanup; exit 0' INT

# Run main function with all arguments
main "$@"
