# Changelog

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
