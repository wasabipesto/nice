# Changelog

## Unreleased

- API: `/status` now reports the build it is running (`build_version`, and `build_commit` when the build was given one) and the process `start_time`, alongside the database pool's occupancy (`pool_connections_active`, `pool_connections_idle`, `pool_connections_max`). The pool numbers come from r2d2's own bookkeeping, so the endpoint still checks out no connection and touches no table. The commit is passed to the docker build as `--build-arg NICE_BUILD_COMMIT=...`, since `.dockerignore` keeps `.git` out of the build context; builds without it report `null`.
- Benchmark: each niceonly GPU scenario now steers the MSD floor for three seconds from the production seed, then holds it for the timed windows and reports it per scenario (`msd_floor`), instead of pinning the whole sweep at the controller's cap. A pinned floor undersold GPU-bound machines by a third (an RTX 3060 with 19 cores measured 3.35e12 pinned against 5.0e12 steered) while a floor moving during a 10 ms window would make the rate depend on the controller's cycle. `NICE_GPU_MSD_FLOOR` still pins. The report's `config.gpu_msd_floor` is replaced by the per-scenario field.
- GPU niceonly: the host pipeline is now continuous across fields. The MSD workers start on the next field while the device is still draining the current one, and the device is fed batches as they arrive, so neither side waits at a field boundary; the client keeps one field started ahead of the one it finishes. The MSD floor is steered by which side is waiting (the dispatch thread blocked pulling descriptors, or blocked handing work to a full device) rather than by per-field timings, starting from 250k. `NICE_GPU_MSD_FLOOR` still pins the floor; `NICE_GPU_FIELDS_IN_FLIGHT=1` restores the field-serial behaviour for comparison and `NICE_GPU_BATCHES_IN_FLIGHT` sets the device queue depth. Per-field rates are now measured as the time since the previous field's result, since neighbouring fields overlap. Submission telemetry for GPU niceonly fields gains a `pipeline` object: the MSD floor, fields and batches in flight, how long the dispatch thread waited on the CPU side and on the device, the device's busy time on the field (CUDA, from per-batch events), descriptor and launch counts — enough to tell a CPU-bound machine from a GPU-bound one.
- GPU niceonly: the MSD filter now starts at blocks of 64 chunks instead of one chunk at a time, so a single analysis can reject 64 chunks at once. Blocks are always a power of two of chunks, which makes the recursion land on chunk boundaries and produce bit-identical descriptors to the chunk start (asserted by a test); only the work changes. On the base-54 regions from the Anvil runs the MSD phase of a 1e12 field went from 0.34 s to 0.18-0.29 s at floor 500k and from 0.49-0.54 s to 0.35-0.45 s at 250k (four cores); where nothing rejects above the chunk it is a wash.
- Report the real GPU name when the `cubecl-cuda` backend is in use. It was reporting the placeholder `cubecl-cuda device 0` (CubeCL's CUDA runtime has no device-name API), and since v3.4.3 made `cubecl-cuda` the backend `--gpu-backend auto` selects for detailed mode on NVIDIA, every detailed benchmark report and `--telemetry` submission from an NVIDIA machine carried that placeholder as `hardware.gpu_model`. The estimator matches offers on that string, so those samples could not match any GPU and fell through to the low-confidence `floor` stage — which the fleet controller refuses to buy on. The name now comes from the CUDA driver (`cuDeviceGetName`), matching what the hand-written `cuda` backend already reports.
- API: serve the detailed `Next` claim strategies (15% of detailed claims at `check_level <= 1`, 4% rechecks at `check_level <= 2`) from pre-claimed in-memory queues, like `Thin` and niceonly already were. The direct `Next` query re-sorts the frontier chunk on every request, so it serialised under load: 14 ms alone but over 100 ms per request with sixteen concurrent clients on a 3M-row model of production. A queue refill is one 22 ms batch per 100 claims. Only `Random` (1%) still claims directly. `/status` reports the new queue sizes.
- API: the detailed-thin queue refill ordered by field id alone, which let the generic plan walk the primary key from id 1 to the frontier chunk (254 ms and 480k rows read per refill on the model; production's frontier is tens of millions of rows deeper). It now orders by chunk then id, the same fix the `Next` claim received, and reads only the frontier chunk (29 ms).

## Nice v3.4.3

- Ship the CUDA runtime headers in the `-gpu` docker image. NVRTC could not find `cuda_runtime.h` in the runtime-only base, so the `cubecl-cuda` backend failed to compile every kernel and `--gpu-backend auto` silently fell through to the hand-written CUDA backend. CI now checks the header is present in the built image.
- GPU niceonly: cap the adaptive MSD floor at half a processing chunk so the controller can no longer ratchet into the no-MSD bypass. The bypass remains available by pinning e.g. `NICE_GPU_MSD_FLOOR=1000000` explicitly.
- GPU niceonly benchmark: pin the MSD floor at the adaptive controller's cap instead of letting it adapt per field in order to be comparable across machines and runs. `NICE_GPU_MSD_FLOOR` still overrides it for floor sweeps. The report's `config` now includes `gpu_msd_floor`.
- GPU niceonly: MSD workers now send descriptors to the dispatch thread in batches (4096 ranges or 256 chunks) instead of one message per chunk.

## Nice v3.4.2

- Fix the `-gpu` docker image failing to start with "version `GLIBC_2.38' not found": the binary is built on Ubuntu 24.04 (glibc 2.39) and the rustls crypto backend that #117 pulled in references glibc 2.38 symbols, but the image's Ubuntu 22.04 base only had 2.35. The base is now `nvidia/cuda:12.8.1-runtime-ubuntu24.04`, and CI runs the binary inside every image it builds before pushing.
- Build the Linux-x86_64 and Linux-aarch64 release binaries inside pinned `cross` containers (Ubuntu 16.04, glibc 2.23) instead of on the GitHub runner, so they run on older hosts (the 3.4.1 binaries required glibc 2.38); CI asserts a glibc 2.28 floor on the artifacts.
- Add opt-in `cubecl-spirv` and `cubecl-metal` build features that route the `cubecl` backend through CubeCL's direct SPIR-V/MSL compilers instead of naga. Fix the kernels' atomic buffers being bound as read-only inputs, which the Metal compiler rejects. The wgpu device name reported in benchmarks and telemetry now includes the shader compiler (`wgpu<wgsl>`, `wgpu<spirv>`, `wgpu<msl>`).
- Add a plane-scoped variant of the CubeCL niceonly compaction queue: each subgroup owns a private queue and drains it without cube-wide barriers. On by default under CubeCL's SPIR-V compiler, where it measures +18-22% on every niceonly scenario on an RX 9070 XT (+11-13% over the previous naga-compiled binary), and under `cubecl-cuda` (+9-18% on an RTX 3090 when the device is the bottleneck); `NICE_CUBECL_PLANE_COMPACT=0|1` overrides. The `gpu` build now includes `cubecl-spirv` so Linux Vulkan users get this path.
- The chunk-scan flavor is now chosen per device: CUDA keeps the 64-bit scan, wgpu keeps the 32-bit split scan (measured 1.4-4.6x faster than the 64-bit scan on AMD RDNA4 and Apple M4). `NICE_CUBECL_WIDE=0|1` forces either for A/B runs.

## Nice v3.4.1

- Add a new sound cross-end residue filter to the CPU and GPU niceonly paths. The MSD interval-domain analysis now returns a certificate of digits that provably occupy high output positions for every n in a surviving range, and the stride iteration skips any residue whose exact low output digits intersect it.
- Raise the CPU MSD recursion floor from 1000 to 8000: with the cross-end filter absorbing the extra candidates a coarse floor lets through, a two-machine whole-field sweep puts the new optimum at 8000.
- Add an explicit no-MSD bypass to the GPU pipeline. At the maximum GPU recursion floor (now one full processing chunk), chunks are shipped as single descriptors with no endpoint analysis, letting a strong device paired with a weak CPU skip host-side MSD filtering entirely. The adaptive floor controller can now discover this mode on its own.
- Compact the cross-end filter's survivors in the CUDA and CubeCL niceonly kernels before checking them. Dispatches whose certificates are all zero (the no-MSD bypass) take the plain path.
- The environment variables `NICE_CUDA_CROSS=0`, `NICE_CUDA_COMPACT=0`, `NICE_CUBECL_CROSS=0`, and `NICE_CUBECL_COMPACT=0` can be set to disable the above behavior for testing and evaluation, but these options will be removed in a future release.
- Segment the API `/estimate` endpoint and python fleet estimator by client version to avoid newer/older clients from swaying the EV calculations.

## Nice v3.4.0

Client features:

- Add new GPU backends `cubecl`, `cubecl-cuda`, and `vulkan` (experimental) for additional device support in GPU-enabled builds. Notably the client can now utilize consumer AMD GPUs and Apple M-series GPUs. No new configuration is needed, the client will automatically detect and use a compatible backend. To manually select a backend use `--gpu-backend`.
- Add WebGPU support to the live WASM web client, which now supports most GPU/OS/browser configurations. (Notable exceptions include AMD cards on linux for most browsers except Firefox Nightly.)
- Pipeline the web client similarly to the native client with a small claim buffer and parallel network calls to keep the backend working at full duty cycle. Add a greedy work queue so an efficiency core does not slow down the whole client.
- Add the fleet controller (`fleet/`): a cron-driven explore/exploit loop over the Vast.ai interruptible market utilizing croudsourced benchmarks
- Rework `--benchmark` into a structured sweep over many bases and windows to fill an adjustable time budget (`--benchmark-secs`, default 10) and output (table by default or json with `--benchmark-json`). Reports hardware info, recorded performance, API latency, scheduler correlation IDs, and a synthetic score. Allows uploading to the coordination server for aggregation (manually or with `--benchmark-upload`).
- Add the reworked benchmark (and synthetic NiceMark score) process to the web client with similar options for work sizing and uploading the score. This client doesn't have access to the CPU details so can't be used for the fleet estimator but can still be used for bragging rights.
- Add opt-in submission telemetry: `--telemetry` attaches hardware, scheduler environment, client config, and client-side processing time to each submission
- Client sub-options now imply their umbrella flag: any explicit `--gpu-*` option enables `--gpu`, and any `--benchmark-*` option enables `--benchmark`. Values set through the `NICE_*` environment variables trigger this also, with boolean `NICE_*` environment variables parsing falsy values (`NICE_GPU=false` or `=0` do not enable the GPU).

Performance:

- Replace the MSD filter's common-prefix duplicate/overlap checks with an interval digit-domain analysis (Hall's theorem). Each near-fixed output position of n² and n³ gets a conservative digit domain, and a range is skipped when the constrained positions cannot all receive distinct digits.
- Deepen the stride table's LSD filter from k=2 to k=3, removing 15-22% of candidates before the nice check, which was made effective by precomputing specific steps of the process
- Raise the CPU MSD recursion floor from 250 to 1000 due to the increased speed of per-digit checks and lower need for MSD prefiltering

Website:

- Start the index page data fetches while the plot library is still downloading, draw each chart as its own data arrives instead of waiting for all five requests, and load d3/Plot as vendored files
- Draw the notable numbers chart from a subset stored in `cache_notable_numbers` instead of every base's full top-10k list
- Rework the web runner's statistics into a session view: fields completed, session totals and rates, best find, a rate-per-field strip chart, and all-time per-browser totals persisted in localStorage
- Add a uniform-random occupancy baseline to the web runner's digit distribution histogram overlay so a field's shape is legible against the expected values

Backend:

- Add `POST /benchmark`: saves community benchmarks from user reports, validating schema and capping report size
- Add `POST /estimate`: predicts per-scenario and blended performance for a hardware configuration from recent benchmark uploads via hierarchical matching (experimental, this endpoint may be removed at any point)
- Add `GET /ping`: a static response for client-side latency measurement
- Make the scheduled jobs incremental and stream the downsampling aggregation to decrease job time and memory usage. A `--full` option restores the previous sweep-everything behavior for use after manual disqualifications.
- Serve `/estimate` from an in-memory benchmark corpus cache instead of re-fetching and re-decoding up to 2000 jsonb reports per request 
- Optimize claim API calls in the server:
  - Scope the detailed `Thin` claim to the proper frontier chunk by checking for claimable fields
  - Scope the detailed `Next` claim to the proper frontier chunk via `chunks.minimum_cl`
  - Make queue refills single-flight and run them on the requesting handler's database connection
  - Thanks to [Janzert](https://github.com/Janzert) for the report and diagnoses
- Add `common/tests/claim_queries.rs`: the claim path is hand-written SQL with positional bindings that the compiler cannot check, so it is now exercised against a real PostgreSQL. Skipped unless `NICE_TEST_DATABASE_URL` is set.

Maintenance:

- Update all dependencies to their latest versions, including major bumps of malachite, reqwest, sysinfo, and rocket_prometheus. This resolves every RustSec advisory fixable within our dependency tree; the one remaining advisory (RUSTSEC-2026-0258, h2 0.3.x) comes via rocket 0.5's hyper 0.14 and has no fixed 0.3.x release
- With reqwest 0.13, default (rustls) builds of the client now verify TLS against the platform trust store instead of bundled webpki roots, and the rustls crypto provider is aws-lc-rs instead of ring. Environments without system CA certificates (bare containers running the static musl binary) now need `ca-certificates` installed; the published docker images already include it.
- Enforce formatting and clippy (all targets, including the GPU feature builds) in CI via a new `just lint` recipe, and fix everything it flagged.
- Remove the unused `docker/` build-container Dockerfiles, superseded by the cross-compiled release workflow

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
