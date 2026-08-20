# Changelog

## Unreleased

Features:

- Add new GPU backends `cubecl`, `cubecl-cuda`, and `vulkan` (experimental) for additional device support in GPU-enabled builds. Notably the client can now utilize consumer AMD GPUs and Apple M-series GPUs. No new configuration is needed, the client will automatically detect and use a compatible backend. To manually select a backend use `--gpu-backend`.
- Add the fleet controller (`fleet/`): a cron-driven explore/exploit loop over the Vast.ai interruptible market utilizing croudsourced benchmarks
- Rework `--benchmark` into a structured sweep over many bases and windows to fill an adjustable time budget (`--benchmark-secs`, default 10). Reports hardware info, recorded performance, API latency, scheduler correlation IDs, and a synthetic score. Allows uploading to the coordination server for aggregation (manually or with `--benchmark-upload`).
- Add opt-in submission telemetry: `--telemetry` attaches hardware, scheduler environment, client config, and client-side processing time to each submission

Performance:

- Replace the MSD filter's common-prefix duplicate/overlap checks with an interval digit-domain analysis (Hall's theorem). Each near-fixed output position of n² and n³ gets a conservative digit domain, and a range is skipped when the constrained positions cannot all receive distinct digits.
- Deepen the stride table's LSD filter from k=2 to k=3, removing 15-22% of candidates before the nice check, which was made effective by precomputing specific steps of the process
- Raise the CPU MSD recursion floor from 250 to 1000 due to the increased speed of per-digit checks and lower need for MSD prefiltering

Backend:

- Add `POST /benchmark`: saves community benchmarks from user reports, validating schema and capping report size
- Add `POST /estimate`: predicts per-scenario and blended performance for a hardware configuration from recent benchmark uploads via hierarchical matching
- Add `GET /ping`: a static response for client-side latency measurement
- Make the scheduled jobs incremental and stream the downsampling aggregation to decrease job time and memory usage. A `--full` option restores the previous sweep-everything behavior for use after manual disqualifications.
- Serve `/estimate` from an in-memory benchmark corpus cache instead of re-fetching and re-decoding up to 2000 jsonb reports per request 
- Optimize claim API calls in the server:
  - Scope the detailed `Thin` claim to the proper frontier chunk by checking for claimable fields
  - Scope the detailed `Next` claim to the proper frontier chunk via `chunks.minimum_cl`
  - Make queue refills single-flight and run them on the requesting handler's database connection
  - Thanks to [Janzert](https://github.com/Janzert) for the report and diagnoses
- Add `common/tests/claim_queries.rs`: the claim path is hand-written SQL with positional bindings that the compiler cannot check, so it is now exercised against a real PostgreSQL. Skipped unless `NICE_TEST_DATABASE_URL` is set.

## Nice v3.3.0

- Fix an off-by-one error in get_base_range_natural which dropped one trailing candidate per base where base % 5 ∈ {2,3,4}. Thanks to [Janzert](https://github.com/Janzert) for reporting this!
- Adapt CPU client processing chunk size from field size to reduce memory usage with larger fields and make TQDM output more readable
- Buffer client claims and overlap the requests that refill them. Thanks to [Janzert](https://github.com/Janzert) for implementing this!
- Refactor the client to utilize the async API and simplify some codepaths
- Remove an unsound MSD x LSD cross-check, which could silently skip valid nice numbers searched in niceonly mode by the CPU client. Submissions from clients 3.2.12-3.1.15 may be annulled while we re-check results. Detailed mode was unaffected.

## Nice v3.2.15

- Fix a bug in the nice-only GPU mode where valid nice numbers would be silently skipped in some bases between 10-25 due to an invalid prefilter configuration. This had no effect on live fields but affected some benchmarks.
- Optimize the nice-only GPU path slightly by disabling the prefilter on bases 41 or higher where warp stats indicate it already runs the full check regardless
- Optimize the nice-only GPU path slightly by computing the square's digits and finding conflicts before calculating the cube
- Optimize the nice-only GPU path slightly by accumulating duplicate flags across a chunk and testing once per chunk instead of branching per digit

## Nice v3.2.14

- Rewrite the CUDA GPU path from scratch. Kernels are now JIT-specialized per base and can be verified with libnvrtc. For detailed search, the GPU handles all processing. Not nice-only search, the CPU handles MSD filtering and the GPU handles bulk nice checks. The handoff between the CPU and GPU is adapted over time but can be overridden with `NICE_GPU_MSD_FLOOR`.
- Add a field queue for the detailed thin strategy, which speeds up the `/detailed/claim` endpoint about 80% of the time

## Nice v3.2.13
- Add additional fields to the database in bases 52, 53, and 54
- Remove dependence on the DEFAULT_FIELD_SIZE constant. The newest fields are larger than this so the server and client are now able to handle them properly. Detailed search claims will be limited to the previous max size (1e9).
- Add recent search progress and overall leaderboard charts to the site
- Add fixed-width compile-time generation to replace div_asign_rem in most cases

## Nice v3.2.12

- Asynchronously submit the previous search field and get the next one while processing to reduce network overhead
- Reuse server connection between claim/submit requests to reduce network overhead
- Implement higher k-values for the experimental LSD filter, fix a bug and add tests
- Implement checks for overlap between the most- and least-significant digits
- Implement a CRT stride filter that allows the process to jump between valid candidates instead of iterating over all of them
- Configure additional trace logging in the library
- Replace `--verbose` and `--quiet` with `--log-level`/`-l` and `--no-progress`/`-n`
- Add support for customizing the number of API retries with `--api-max-retries`
- Update base bounds script
- Update filter effectiveness script

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
