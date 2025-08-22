# Nice Numbers Web Client

A WebAssembly-based browser client for the Nice Numbers distributed computing project. This client allows you to contribute to the search for "nice numbers" (square-cube pandigitals) directly from your web browser.

## What are Nice Numbers?

Nice numbers are square-cube pandigitals - numbers where the digits in their square and cube contain all possible digit values with no repeats. Currently, 69 (in base 10) is the only known nice number, but mathematicians believe there should be more in other bases!

## Features

- 🌐 **Browser-based**: No installation required, runs directly in your web browser
- ⚡ **WebAssembly powered**: Uses Rust compiled to WASM for high performance
- 🔧 **Two search modes**:
  - **Nice-only mode**: Fast search that only finds 100% nice numbers
  - **Detailed mode**: Slower search that collects comprehensive statistics
- 🧪 **Offline testing**: Built-in benchmark mode for testing without server connection
- 📊 **Real-time progress**: Live updates on processing status and results

## Prerequisites

- **Rust toolchain** (for building)
- **wasm-pack** (for WebAssembly compilation)
- **HTTP server** (for serving the client - required due to WASM security restrictions)

### Installing Dependencies

1. Install Rust from [rustup.rs](https://rustup.rs/)

2. Install wasm-pack:
```bash
curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
```

## Building

1. Navigate to the web-client directory:
```bash
cd nice/web-client
```

2. Run the build script:
```bash
chmod +x build.sh
./build.sh
```

Or build manually:
```bash
wasm-pack build --target web --out-dir pkg
```

## Running

After building, you need to serve the files with an HTTP server (required for WASM modules):

### Option 1: Python
```bash
# Python 3
python3 -m http.server 8000

# Python 2
python -m SimpleHTTPServer 8000
```

### Option 2: Node.js
```bash
npx serve .
```

### Option 3: PHP
```bash
php -S localhost:8000
```

### Option 4: Any other HTTP server
Point your favorite HTTP server to serve the `web-client` directory.

Then open your browser and navigate to `http://localhost:8000`

## Usage

1. **Configure settings**:
   - Enter your username (or leave as "anonymous")
   - Choose search mode (Nice-only is faster, Detailed provides more data)
   - Set API URL (default: https://api.nicenumbers.net)
   - Select Live mode to contribute, or Offline mode for testing

2. **Start processing**:
   - Click "Start Processing" to begin
   - The client will automatically request work from the server
   - Processing happens in your browser using WebAssembly
   - Results are automatically submitted back to the server

3. **Monitor progress**:
   - Watch real-time status updates
   - View processing statistics and any nice numbers found
   - Stop processing at any time

## Search Modes

### Nice-only Mode (Recommended)
- **Fast**: Optimized for speed, typically 20x faster than detailed mode
- **Purpose**: Finds only 100% nice numbers (uses all digits exactly once)
- **Best for**: Most users contributing to the search

### Detailed Mode
- **Slower**: Comprehensive analysis of all numbers in the range
- **Purpose**: Collects statistics on "niceness" distribution for research
- **Best for**: Users contributing to mathematical analysis and research

## Architecture

The web client consists of:

- **Rust/WASM core** (`src/lib.rs`): High-performance number processing
- **JavaScript interface** (`index.html`): Web UI and server communication
- **WebAssembly bridge**: Seamless integration between Rust and JavaScript

Key functions exposed to JavaScript:
- `process_niceonly()`: Fast nice number search
- `process_detailed()`: Comprehensive analysis
- `get_benchmark_field()`: Offline testing data

## Development

### Project Structure
```
web-client/
├── src/
│   └── lib.rs          # Main Rust/WASM implementation
├── pkg/                # Generated WASM files (after build)
├── index.html          # Web interface
├── build.sh           # Build script
├── Cargo.toml         # Rust dependencies
└── README.md          # This file
```

### Key Dependencies
- `wasm-bindgen`: Rust-JavaScript interop
- `malachite`: Arbitrary precision arithmetic
- `serde`: JSON serialization
- `web-sys`: Browser APIs

### Building for Different Targets
The current build targets the web with ES modules. To build for different targets:

```bash
# Web (ES modules) - default
wasm-pack build --target web

# Node.js
wasm-pack build --target nodejs

# Bundler (webpack, etc.)
wasm-pack build --target bundler
```

## Contributing

This web client is part of the larger Nice Numbers project. To contribute:

1. Test the client thoroughly in different browsers
2. Report any bugs or performance issues
3. Suggest UI improvements
4. Help optimize the WASM performance
5. Add new features (with appropriate tests)

## Security Considerations

- The client runs entirely in your browser - no external executables
- All network communication is with the official Nice Numbers API
- Processing happens locally using WebAssembly
- No personal data is collected beyond the username you provide

## Performance

Typical performance on modern browsers:
- **Nice-only mode**: ~1M numbers/second
- **Detailed mode**: ~50K numbers/second

Performance varies based on:
- Browser and JavaScript engine
- CPU speed and available cores
- Base number being searched
- Range size assigned by server

## Troubleshooting

### WASM fails to load
- Ensure you're serving files via HTTP/HTTPS (not file://)
- Check browser console for specific error messages
- Try a different browser (Chrome, Firefox, Safari, Edge)

### Slow performance
- Try Nice-only mode instead of Detailed mode
- Close other browser tabs/applications
- Check if your browser supports WebAssembly (all modern browsers do)

### Connection errors
- Verify the API URL is correct
- Check your internet connection
- Try offline/benchmark mode for testing

## License

This project is part of the Nice Numbers distributed computing project. See the main project LICENSE file for details.