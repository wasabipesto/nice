# Changelog

## Unreleased

- Asynchronously submit the previous search field and get the next one while processing to reduce network overhead
- Reuse server connection client between claim/submit requests
- Fix a bug in the experimental LSD filter that gave false negatives, added tests and enabled it for nice-only processing
- Configure additional trace logging in the library
- Replace `--verbose` and `--quiet` with `--log-level`/`-l` and `--no-progress`/`-n`
- Add support for customizing the number of API retries with `--api-max-retries`

## Nice v3.2.11

- Implement some coarse but massive optimizations to nice-only processing based on patterns in the most and least significant digits of each range. The exact amount varies by search range but in the current area of interest it is about 2.5x as fast.
- Implement an in-memory queue for nice-only claims so the server can keep up with the increased processing speed. This takes nice-only claim endpoint times from 90-100ms to 3-5ms.
- Add a Prometheus exporter to monitor the response times on each API endpoint
- Fix an issue where the largest numbers (instead of nicest) are preserved during downsampling
- Decouple detailed chunk processing size from rayon chunk processing size
- Start using proper rust logging systems such as env_logger and test-log

## Nice v3.2.10

- Add validate function to client and server to confirm that new results are consistent with past submissions
- Add experimental support for GPU acceleration with CUDA

## Nice v3.2.9

- Increase maximum retry attempts from 6 (max delay 32 seconds) to 10 (max delay 512 seconds)
- Add some checks to ensure that git tag pushing is done cleanly

## Nice v3.2.8

- Add some release profile configuration options for a little more performance
- Add new claim strategy "Thin" which gets a random unchecked field in the next chunk with under a certain percent checked

## Nice v3.2.7

- Allow the client to retry on request errors and add a bit more logging in case failures continue
- Add CORS headers directly to the API instead of through CDN
- Add some more logging in the server for an edge case
- Drastically improve scheduled job downsampling performance
- Fix WASM builds by gating the rand crate behind the database feature
- Update dependencies

## Nice v3.2.6

- Update web chart formatting
- Add additional indexes to the database
- Add database connection pooling to the API server
- Add better logging, tracing, and error handling to the API server
- Drastically speed up claims via the API
- Show API server error responses in the client
- Add CI builds for 32-bit Linux

## Nice v3.2.5

- Bump to force CI test release

## Nice v3.2.4

- Additional retries and exponential backoff in client
- Faster docker builds in CI

## Nice v3.2.3

- Update dependencies
- Bump rust edition to 2024
- Migrate openssl-tls to rustls-tls
- Re-enable multi-platform builds and testing CI
- Allow setting client options via environment variables
- Add client dockerfile and publish images to GHCR

## Nice v3.2.1

- Fixes a bug where the native client would crash upon beginning the second iteration of a --repeat loop due to rayon's thread pool already being initialized.

## Nice v3.2.0

- WebAssembly module, integrated into the search page found at https://nicenumbers.net/search/
- Native client progress bars and parallelization. Both are enabled by default, but you can silence the progress bar with --quiet and you can customize the number of threads with --threads.
