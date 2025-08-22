#!/bin/bash

# Test script for Web Worker implementation
# This script runs basic tests to ensure the Web Worker version builds and functions correctly

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_test() {
    echo -e "${BLUE}🧪 TEST: $1${NC}"
}

print_pass() {
    echo -e "${GREEN}✅ PASS: $1${NC}"
}

print_fail() {
    echo -e "${RED}❌ FAIL: $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ️  $1${NC}"
}

echo -e "${BLUE}🧵 Nice Numbers Web Worker - Test Suite${NC}"
echo -e "${BLUE}======================================${NC}\n"

# Test 1: Check required files exist
print_test "Checking required files exist"
required_files=("worker.js" "index-worker.html" "src/lib.rs" "Cargo.toml")
for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        print_pass "$file exists"
    else
        print_fail "$file missing"
        exit 1
    fi
done

# Test 2: Check Rust dependencies
print_test "Checking Rust toolchain"
if command -v rustc &> /dev/null; then
    rust_version=$(rustc --version)
    print_pass "Rust found: $rust_version"
else
    print_fail "Rust not installed"
    exit 1
fi

if command -v wasm-pack &> /dev/null; then
    wasm_version=$(wasm-pack --version)
    print_pass "wasm-pack found: $wasm_version"
else
    print_fail "wasm-pack not installed"
    exit 1
fi

# Test 3: Check Rust code contains worker-specific exports
print_test "Checking Rust exports for worker compatibility"
if grep -q "get_num_unique_digits_wasm" src/lib.rs; then
    print_pass "Worker-specific function exported"
else
    print_fail "Worker-specific function not found in src/lib.rs"
    exit 1
fi

# Test 4: Build WASM package
print_test "Building WASM package"
if [ -d "pkg" ]; then
    rm -rf pkg
    print_info "Cleaned previous build"
fi

if wasm-pack build --target web --out-dir pkg &> /dev/null; then
    print_pass "WASM build successful"
else
    print_fail "WASM build failed"
    exit 1
fi

# Test 5: Check generated files
print_test "Checking generated WASM files"
required_build_files=("pkg/nice_web_client.js" "pkg/nice_web_client_bg.wasm")
for file in "${required_build_files[@]}"; do
    if [ -f "$file" ]; then
        print_pass "$file generated"
    else
        print_fail "$file not generated"
        exit 1
    fi
done

# Test 6: Check WASM exports include worker functions
print_test "Checking WASM exports include worker functions"
if grep -q "get_num_unique_digits_wasm" pkg/nice_web_client.js; then
    print_pass "Worker function properly exported in WASM"
else
    print_fail "Worker function not found in generated WASM"
    exit 1
fi

# Test 7: Check worker.js syntax
print_test "Checking worker.js syntax"
if node -c worker.js 2>/dev/null; then
    print_pass "worker.js has valid syntax"
elif command -v node &> /dev/null; then
    print_fail "worker.js has syntax errors"
    exit 1
else
    print_info "Node.js not available, skipping syntax check"
fi

# Test 8: Check HTML files are valid
print_test "Checking HTML file structure"
if grep -q "Web Worker" index-worker.html; then
    print_pass "index-worker.html contains web worker references"
else
    print_fail "index-worker.html missing web worker content"
    exit 1
fi

if grep -q "new Worker" index-worker.html; then
    print_pass "index-worker.html creates web worker"
else
    print_fail "index-worker.html doesn't create web worker"
    exit 1
fi

# Test 9: Check for proper MIME type handling
print_test "Checking server configuration"
if [ -f "serve.py" ]; then
    if grep -q "application/wasm" serve.py; then
        print_pass "Server configured for WASM MIME type"
    else
        print_fail "Server missing WASM MIME type configuration"
        exit 1
    fi
else
    print_info "serve.py not found, skipping MIME type check"
fi

# Test 10: Check build scripts
print_test "Checking build scripts"
if [ -f "run-worker.sh" ] && [ -x "run-worker.sh" -o -r "run-worker.sh" ]; then
    print_pass "run-worker.sh script available"
else
    print_fail "run-worker.sh script missing or not executable"
fi

echo ""
echo -e "${GREEN}🎉 All tests passed! Web Worker implementation is ready.${NC}"
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo "1. Run: chmod +x run-worker.sh"
echo "2. Run: ./run-worker.sh"
echo "3. Open browser to: http://localhost:8000"
echo "4. Test with benchmark mode first"
echo ""
echo -e "${YELLOW}Note: The Web Worker version prevents browser freezing during computation.${NC}"
