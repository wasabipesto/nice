#!/bin/bash

# Nice Numbers Web Client - Setup and Run Script
# This script handles the complete setup and execution of the web client

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
    echo -e "\n${BLUE}🔢 Nice Numbers Web Client${NC}"
    echo -e "${BLUE}=================================${NC}\n"
}

# Check if we're in the right directory
check_directory() {
    if [ ! -f "Cargo.toml" ] || [ ! -f "index.html" ]; then
        print_error "Not in the web-client directory!"
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

# Check Web Worker files
check_worker_files() {
    if [ ! -f "worker.js" ]; then
        print_error "worker.js not found!"
        print_info "The Web Worker script is required for this client"
        exit 1
    fi

    print_success "Web Worker files found"
}

# Build the WASM package
build_wasm() {
    print_info "Building WebAssembly package..."

    # Clean previous build
    if [ -d "pkg" ]; then
        rm -rf pkg
        print_info "Cleaned previous build"
    fi

    # Build with wasm-pack
    wasm-pack build --target web --out-dir pkg

    if [ $? -eq 0 ] && [ -d "pkg" ]; then
        print_success "WASM build completed successfully!"

        # Check if worker functions are properly exported
        if grep -q "get_num_unique_digits_wasm" pkg/nice_web_client.js; then
            print_success "Web Worker functions properly exported"
        else
            print_warning "Web Worker functions might not be exported correctly"
        fi

        # List generated files
        print_info "Generated files:"
        ls -la pkg/ | grep -E '\.(js|wasm|ts)$' || true
    else
        print_error "WASM build failed!"
        exit 1
    fi
}

# Start the development server
start_server() {
    local port=${1:-8000}

    print_info "Starting development server on port $port..."
    print_info "🧵 This client uses Web Workers to prevent browser freezing"

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
        print_info "- Then open: http://localhost:$port"
        exit 1
    fi
}

# Display usage information
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --build-only     Build WASM package only (don't start server)"
    echo "  --serve-only     Start server only (skip build)"
    echo "  --port PORT      Use specific port (default: 8000)"
    echo "  --help           Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                    # Build and serve on port 8000"
    echo "  $0 --port 3000       # Build and serve on port 3000"
    echo "  $0 --build-only      # Just build, don't serve"
    echo "  $0 --serve-only      # Just serve (assumes already built)"
}

# Main script logic
main() {
    local build_only=false
    local serve_only=false
    local port=8000

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
        echo ""
        print_success "Setup complete! Starting Web Worker client..."
        print_info "🧵 Web Workers prevent browser freezing during computation"
        print_info "Open your browser and navigate to: http://localhost:$port"
        print_warning "Press Ctrl+C to stop the server"
        echo ""

        # Give user a moment to read the instructions
        sleep 2

        start_server $port
    else
        print_success "Build completed! To serve the files:"
        print_info "Run: $0 --serve-only --port $port"
        print_info "Or use any HTTP server to serve this directory"
    fi
}

# Handle Ctrl+C gracefully
trap 'echo -e "\n${YELLOW}🛑 Stopping...${NC}"; exit 0' INT

# Run main function with all arguments
main "$@"
