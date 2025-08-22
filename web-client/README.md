# Nice Numbers - Browser Client

A WebAssembly-powered browser client for the Nice Numbers distributed computing project. This client allows you to contribute to the search for "nice numbers" (square-cube pandigitals) directly from your web browser using Web Workers for responsive performance.

## What are Nice Numbers?

Nice numbers are square-cube pandigitals - numbers where the digits in their square and cube contain all possible digit values with no repeats. Currently, 69 (in base 10) is the only known nice number, but mathematicians believe there should be more in other bases!

## Features

- 🌐 **Browser-based**: No installation required, runs directly in your web browser
- ⚡ **WebAssembly powered**: Uses Rust compiled to WASM for high performance
- 🧵 **Web Worker architecture**: Non-blocking computation keeps your browser responsive
- 🔧 **Two search modes**:
  - **Nice-only mode**: Fast search that only finds 100% nice numbers
  - **Detailed mode**: Slower search that collects comprehensive statistics
- 🧪 **Offline testing**: Built-in benchmark mode for testing without server connection
- 📊 **Real-time progress**: Live updates on processing status without UI freezing

## Quick Start

1. **Install dependencies:**
   ```bash
   # Install Rust (if not already installed)
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   
   # Install wasm-pack (if not already installed)
   curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
   ```

2. **Build and run:**
   ```bash
   cd nice/web-client
   chmod +x run.sh
   ./run.sh
   ```

3. **Open browser:**
   - Navigate to `http://localhost:8000`
   - Configure your settings and click "Start Processing"

## Architecture

### Web Worker Design
The client uses a Web Worker architecture to prevent browser freezing:

```
Main Thread: [UI & Controls] ←→ [PostMessage API] ←→ [Web Worker: WASM Computation]
```

**Benefits:**
- UI remains responsive during computation
- Real-time progress updates every second
- Immediate response to stop requests
- Better user experience for long-running tasks

### Key Components
- **`index.html`** - Main interface and Web Worker management
- **`worker.js`** - Background computation thread
- **`src/lib.rs`** - Rust/WASM core with digit counting algorithms
- **`run.sh`** - Build and serve script

## Usage

### Configuration
- **Username**: Your contributor name (defaults to "anonymous")
- **Processing Mode**: 
  - Nice-only: Fast mode, finds only 100% nice numbers
  - Detailed: Slower mode, includes statistical analysis
- **Server URL**: API endpoint (default: https://api.nicenumbers.net)
- **Test Mode**: 
  - Online: Connects to server for real work
  - Offline: Uses benchmark data for testing

### Processing
1. Click "Start Processing" to begin
2. Monitor real-time progress updates
3. Results are automatically submitted to server (online mode)
4. Processing continues automatically until stopped
5. Click "Stop Processing" to halt immediately

## Performance

### Typical Rates (Modern Browser)
- **Nice-only mode**: ~1M numbers/second
- **Detailed mode**: ~50K numbers/second
- **Progress updates**: Every 1 second with minimal overhead
- **Stop response**: <100ms from user interaction

### Browser Compatibility
- **Chrome/Chromium**: Excellent performance
- **Firefox**: Excellent performance
- **Safari**: Good performance
- **Edge**: Excellent performance

**Requirements:**
- Web Workers support (all modern browsers)
- WebAssembly support (all modern browsers)
- HTTP/HTTPS serving (required for WASM security)

## Development

### Project Structure
```
web-client/
├── src/lib.rs          # Rust/WASM implementation
├── index.html          # Main Web Worker interface
├── worker.js           # Web Worker background script
├── run.sh              # Build and serve script
├── serve.py            # Development server with WASM support
├── pkg/                # Generated WASM files (after build)
└── Cargo.toml          # Rust dependencies
```

### Key Dependencies
- `wasm-bindgen`: Rust-JavaScript interop
- `malachite`: Arbitrary precision arithmetic for large number handling
- `web-sys`: Browser API bindings

### Manual Build
```bash
# Build WASM package
wasm-pack build --target web --out-dir pkg

# Serve files (required for WASM)
python3 -m http.server 8000

# Open browser to http://localhost:8000
```

### Customization
Modify computation parameters in `worker.js`:
```javascript
const progressUpdateInterval = 1000; // Progress update frequency
const chunkSize = BigInt(1000);       // Numbers processed per chunk
```

## Troubleshooting

### Common Issues

**WASM fails to load**
- Ensure files are served via HTTP/HTTPS (not `file://`)
- Check browser console for specific errors
- Verify `pkg/` directory exists and contains `.wasm` files

**Browser freezing**
- This shouldn't happen with the Web Worker version
- Check that Web Workers are supported in your browser
- Monitor browser console for Web Worker errors

**Slow performance**
- Try Nice-only mode instead of Detailed mode
- Close unnecessary browser tabs
- Check if other applications are using CPU

**Connection errors**
- Verify API URL is correct and accessible
- Test with offline/benchmark mode first
- Check internet connection

**Web Worker not loading**
- Ensure `worker.js` exists and is served correctly
- Check browser console for worker creation errors
- Verify WASM files are built and available

## Security & Privacy

- Runs entirely in your browser - no external executables
- Only communicates with the official Nice Numbers API
- Processing happens locally using WebAssembly
- No personal data collected beyond provided username
- Open source - audit the code yourself

## Contributing

Ways to help:
1. **Compute**: Run the client to contribute processing power
2. **Test**: Try different browsers and report issues
3. **Optimize**: Improve WASM performance or UI/UX
4. **Document**: Help improve documentation and guides

## License

This project is part of the Nice Numbers distributed computing project. See the main project repository for license details.

---

**Ready to contribute?** Run `./run.sh` and start helping discover new nice numbers!