# Changelog

## Unreleased

- Time out API database statements after 30 seconds, so one stalled claim can no longer block detailed claims and the scheduled jobs ([#238](https://github.com/wasabipesto/nice/pull/238)).
- Let the `Next` queue's cursor read past longer finished stretches before falling back to a full frontier search ([#238](https://github.com/wasabipesto/nice/pull/238)).

## Nice v3.4.7

Performance gains, combined about +30-50% for nice-only and 700-1000% for detailed:

- Speed up the detailed mode by updating the digits of each number's square and cube with additions from the previous number's instead of dividing them out of each one independently ([#221](https://github.com/wasabipesto/nice/pull/221)). On GPU, this is implemented by stepping each thread through a run of consecutive numbers with digit-array additions ([#225](https://github.com/wasabipesto/nice/pull/225), [#230](https://github.com/wasabipesto/nice/pull/230)). Also included in the browser client ([#227](https://github.com/wasabipesto/nice/pull/227), [#233](https://github.com/wasabipesto/nice/pull/233)).
- Speed up GPU detailed mode by copying back only the near-misses found, instead of the whole 20 MB buffer after every field ([#224](https://github.com/wasabipesto/nice/pull/224)).
- Speed up the nice-only overlap join by dropping, before they are paired with any prefix, the bottom residues whose digits clash with the ones every number of the field shares ([#215](https://github.com/wasabipesto/nice/pull/215)).
- Speed up the GPU overlap join by tuning the number of entries per thread: two entries for bases up to 60 and six from 62+ ([#205](https://github.com/wasabipesto/nice/pull/205), [#216](https://github.com/wasabipesto/nice/pull/216)).
- Speed up the CPU overlap join with an AVX-512 scan (where the CPU has it) and a top certificate built from one endpoint's digits instead of both, with half the 256-bit divisions ([#204](https://github.com/wasabipesto/nice/pull/204)).
- Size the overlap join's working memory to the available memory on the GPU (between 1-4 GiB) so a production-sized field at bases 60-64 can run as a single slice instead of 5+ ([#206](https://github.com/wasabipesto/nice/pull/206)).
- Reduce the number of survivors after the overlap join's prefix certificate filtering. At the first position where an interval's powers differ, the digit's range of possible values can rule the interval out or fix one more digit ([#203](https://github.com/wasabipesto/nice/pull/203)).

Remove some experimental and development-only code:

- Remove the `NICE_CUDA_CROSS`, `NICE_CUDA_COMPACT`, `NICE_CUBECL_CROSS` and `NICE_CUBECL_COMPACT` environment variables; the cross-end filter and its compaction now always run where they apply ([#191](https://github.com/wasabipesto/nice/pull/191)).
- Remove the `NICE_GPU_FIELDS_IN_FLIGHT` and `NICE_GPU_BATCHES_IN_FLIGHT` environment variables; the GPU nice-only pipeline keeps its defaults of 2 fields and 16 batches in flight ([#192](https://github.com/wasabipesto/nice/pull/192)).
- Remove the experimental hand-written Vulkan backend, which was unmaintained is now superseded entirely by the `cubecl` backend ([#193](https://github.com/wasabipesto/nice/pull/193)).

Address some Apple-specific correctness and runtime issues:

- Keep the overlap join's GPU top certificates clear of a constant division that Apple's Metal compiler potentially miscompiles ([#228](https://github.com/wasabipesto/nice/pull/228)). Bases 57-64 are not affected.
- Keep the GPU nice-only prefilter at bases 40 and below clear of a constant division that Apple's Metal compiler potentially miscompiles, as a precaution ([#229](https://github.com/wasabipesto/nice/pull/229)).
- Start the overlap join within 512 MiB layouts on the paravirtual GPU of macOS virtual machines, which hangs on larger ones instead of refusing them ([#218](https://github.com/wasabipesto/nice/pull/218)).

Smooth out some API server latency: 

- Update the leaderboard and activity chart caches from each run's new submissions instead of re-reading every submission every hour ([#201](https://github.com/wasabipesto/nice/pull/201)).
- Use a server-side cursor to refill the `Next` queue instead of re-reading the frontier chunk ([#226](https://github.com/wasabipesto/nice/pull/226)).
- Wait for a refill in flight instead of piling direct claims onto it ([#226](https://github.com/wasabipesto/nice/pull/226)).
- Lock claimed fields so that a slow claim cannot hold up `/submit` ([#226](https://github.com/wasabipesto/nice/pull/226)).

Miscellaneous:

- Raise the largest detailed field permitted by the server from 1e9 to 1e10 numbers wide so detailed mode can continue at base 52 once base 50's fields are done ([#223](https://github.com/wasabipesto/nice/pull/223)).
- Use the GPU-style progress bar on CPU also, with consistent log lines for claim and submit actions and the progress bar pinned to the bottom of the terminal ([#202](https://github.com/wasabipesto/nice/pull/202)).
- Drop the priority of the hand-written `cuda` option in the `--gpu-backed auto niceonly` list; you can still select it manually with `--gpu-backend cuda` ([#207](https://github.com/wasabipesto/nice/pull/207)).
- Utilize the nice-only overlap join from CubeCL's path when `--gpu-backend cuda` is selected manually ([#190](https://github.com/wasabipesto/nice/pull/190)).
- Make the fleet require the GPU image's CUDA version on every offer search (`min_cuda_version`, default 12.8), even when a config's `offer_query` asks for less ([#189](https://github.com/wasabipesto/nice/pull/189)).
- Bump dependencies ([#195](https://github.com/wasabipesto/nice/pull/195), [#196](https://github.com/wasabipesto/nice/pull/196), [#197](https://github.com/wasabipesto/nice/pull/197), [#211](https://github.com/wasabipesto/nice/pull/211), [#212](https://github.com/wasabipesto/nice/pull/212), [#213](https://github.com/wasabipesto/nice/pull/213), [#214](https://github.com/wasabipesto/nice/pull/214), [#219](https://github.com/wasabipesto/nice/pull/219), [#220](https://github.com/wasabipesto/nice/pull/220))

## Nice v3.4.6

- Add an overlap join as the primary nice-only path for large fields. It skips much of the existing filter work by joining an overlapping list of the top-digit prefixes and bottom-digit residues which eliminates more candidates with the same underlying strategies. When the device or backend doesn't support this path it falls back to the existing stride implementation. Major thanks to [danstoyell](https://github.com/danstoyell) for the report and prototype ([#178](https://github.com/wasabipesto/nice/pull/178)).
- Add an affine middle-digit filter to the CPU and CUDA nice-only stride paths. It uses an affine map to determine the rightmost digits 3-5 of the square and cube, then compares them with the residue's low digits and the range's high certificate to remove additional candidates ([#166](https://github.com/wasabipesto/nice/pull/166), [#168](https://github.com/wasabipesto/nice/pull/168)).
- Reduce the MSD interval analysis cost per node by utilizing wide division for endpoint digit extraction and masked recursion to reduce the number of total split computations ([#167](https://github.com/wasabipesto/nice/pull/167)).
- Split the CPU nice-only stride walk into two phases. Phase 1 streams the gap and low-mask tables and compacts the survivor list into per-thread buffers with no data-dependent branch, Phase 2 runs the affine middle-digit filter and the seeded nice check on the survivors only ([#169](https://github.com/wasabipesto/nice/pull/169)).
- Make the fleet controller price offers with the same client version it launches. By default `client_version: "auto"` follows the newest release image in the registry and uses that version for both pricing estimates and launching ([#185](https://github.com/wasabipesto/nice/pull/185)).
- Record fleet explore instances that finish their benchmarks and delete themselves as retired instead of preempted, billed up to their last upload ([#186](https://github.com/wasabipesto/nice/pull/186)).
- Restrict the fleet to hosts whose driver supports CUDA 12.8, the version the GPU image is built on ([#187](https://github.com/wasabipesto/nice/pull/187)).
- Warn when an NVIDIA driver is present but CUDA fails to initialize, and list each backend's failure when no GPU backend is usable ([#187](https://github.com/wasabipesto/nice/pull/187)).
- Update `--threads 0` (and `NICE_THREADS=0`) to explicitly spawn as many threads as the host has CPU cores instead of falling back to edge case behavior. `RAYON_NUM_THREADS` no longer influences the client. Omitting this option still defaults to 4 threads ([#153](https://github.com/wasabipesto/nice/pull/153)).
- Lower the CPU MSD recursion floor from 8000 to 4000 since both node checks and survivor checks are cheaper ([#167](https://github.com/wasabipesto/nice/pull/167)).
- Record the build commit SHA alongside the version for benchmarks, telemetry, and api `/status` ([#152](https://github.com/wasabipesto/nice/pull/152)).
- Show a progress bar on GPU runs, like the CPU path has ([#154](https://github.com/wasabipesto/nice/pull/154)).
- Update the nice-only benchmark scenarios to use larger fields in bases 58, 60 and 62 so that they can properly measure the overlap join ([#184](https://github.com/wasabipesto/nice/pull/184)).
- Re-pin the NiceMark reference rates, including the new nice-only fields ([#184](https://github.com/wasabipesto/nice/pull/184)).
- Add a test for the fixed-width arithmetic's "n³ fits in 256 bits" precondition ([#158](https://github.com/wasabipesto/nice/pull/158)).
- Add a brute-force oracle test for the MSD filter's Hall check ([#157](https://github.com/wasabipesto/nice/pull/157)).
- Bump dependency versions, including minimum Rust version 1.99 ([#162](https://github.com/wasabipesto/nice/pull/162), [#163](https://github.com/wasabipesto/nice/pull/163), [#165](https://github.com/wasabipesto/nice/pull/165), [#171](https://github.com/wasabipesto/nice/pull/171), [#172](https://github.com/wasabipesto/nice/pull/172), [#179](https://github.com/wasabipesto/nice/pull/179), [#180](https://github.com/wasabipesto/nice/pull/180), [#181](https://github.com/wasabipesto/nice/pull/181), [#182](https://github.com/wasabipesto/nice/pull/182), [#183](https://github.com/wasabipesto/nice/pull/183)).

## Nice v3.4.5

- Fix NVIDIA niceonly fields failing with `CUDA_ERROR_INVALID_HANDLE` at the end of the field: the per-batch events used to measure device busy time were created with timing disabled, which `cuEventElapsedTime` rejects. Busy-time events are now timing-capable, a timing failure drops `device_busy_secs` instead of failing, and a new GPU test runs a field through the pipeline and checks the busy time comes back ([#143](https://github.com/wasabipesto/nice/pull/143)).
- Take `cubecl-spirv` out of the `gpu` umbrella. On macOS it needs the Vulkan SDK (`VULKAN_SDK` set) just to build, for a compiler that never runs there, which made `--features gpu` fail on a stock mac. `gpu` now builds on every platform with no GPU libraries installed. `cubecl-spirv` is opt-in (`--features gpu,cubecl-spirv`) and the `-gpu` docker image is still built with it, so Linux Vulkan users keep plane-scoped compaction ([#146](https://github.com/wasabipesto/nice/pull/146)).
- Fix `--features vulkan` on its own not compiling: the client's queued-field path was gated on every GPU feature, so with only Vulkan it was a match whose single arm was `unreachable!`, with no type to infer (E0282). It is now gated on the backends that queue ([#150](https://github.com/wasabipesto/nice/pull/150)).
- Under `--gpu-backend auto`, a CUDA-family backend that is skipped because the NVIDIA driver loaded but the CUDA toolkit (NVRTC) is missing now logs a warning saying so, since that machine could be running the faster backend. A machine with no NVIDIA driver at all still falls through quietly. The fatal "no usable GPU backend" message now also lists the compiled-in shader compilers (`cubecl-spirv`, `cubecl-metal`) ([#149](https://github.com/wasabipesto/nice/pull/149)).
- Print warnings up front about specific feature combinations: `cubecl-hip` without `hipconfig` on PATH and `openssl-tls` together with `rustls-tls` ([#147](https://github.com/wasabipesto/nice/pull/147)).

## Nice v3.4.4

- Make the GPU niceonly pipeline hand off results from the CPU MSD filter on an adaptive time basis instead of by size. This should ensure that the GPU and CPU are both utilized to their max. Times are recorded and saved in telemetry, if enabled. See environment knobs `NICE_GPU_MSD_FLOOR`, `NICE_GPU_FIELDS_IN_FLIGHT`, and `NICE_GPU_BATCHES_IN_FLIGHT` for control, though these may be removed in the future ([#140](https://github.com/wasabipesto/nice/pull/140)).
- Let the benchmark warm up the MSD tuning for three seconds before locking in a single value for the actual run ([#141](https://github.com/wasabipesto/nice/pull/141))
- Start the MSD filter at 1/64 of the field size instead of a hardcoded chunk value in order to decrease the amount of upfront cost when CPU is relatively weak and invalid ranges are large ([#139](https://github.com/wasabipesto/nice/pull/139))
- Report the real GPU name when the `cubecl-cuda` backend is in use (previously a placeholder `cubecl-cuda device 0`, now uses`cuDeviceGetName`) ([#137](https://github.com/wasabipesto/nice/pull/137))
- Serve the detailed `Next` claim strategy from pre-claimed in-memory queues, like `Thin` and niceonly already were. Only `Random` (1%) still claims directly. `/status` reports the new queue sizes ([#138](https://github.com/wasabipesto/nice/pull/138)).
- Add uptime, pool status and build version to the status endpoint

## Nice v3.4.3

- Ship the CUDA runtime headers in the `-gpu` docker image. NVRTC could not find `cuda_runtime.h` in the runtime-only base, so the `cubecl-cuda` backend failed to compile every kernel and `--gpu-backend auto` silently fell through to the hand-written CUDA backend. CI now checks the header is present in the built image ([#133](https://github.com/wasabipesto/nice/pull/133)).
- GPU niceonly: cap the adaptive MSD floor at half a processing chunk so the controller can no longer ratchet into the no-MSD bypass. The bypass remains available by pinning e.g. `NICE_GPU_MSD_FLOOR=1000000` explicitly ([#135](https://github.com/wasabipesto/nice/pull/135)).
- GPU niceonly benchmark: pin the MSD floor at the adaptive controller's cap instead of letting it adapt per field in order to be comparable across machines and runs. `NICE_GPU_MSD_FLOOR` still overrides it for floor sweeps. The report's `config` now includes `gpu_msd_floor` ([#136](https://github.com/wasabipesto/nice/pull/136)).
- GPU niceonly: MSD workers now send descriptors to the dispatch thread in batches (4096 ranges or 256 chunks) instead of one message per chunk ([#134](https://github.com/wasabipesto/nice/pull/134)).

## Nice v3.4.2

- Fix the `-gpu` docker image failing to start with "version `GLIBC_2.38' not found": the binary is built on Ubuntu 24.04 (glibc 2.39) and the rustls crypto backend that #117 pulled in references glibc 2.38 symbols, but the image's Ubuntu 22.04 base only had 2.35. The base is now `nvidia/cuda:12.8.1-runtime-ubuntu24.04`, and CI runs the binary inside every image it builds before pushing ([#129](https://github.com/wasabipesto/nice/pull/129)).
- Build the Linux-x86_64 and Linux-aarch64 release binaries inside pinned `cross` containers (Ubuntu 16.04, glibc 2.23) instead of on the GitHub runner, so they run on older hosts (the 3.4.1 binaries required glibc 2.38); CI asserts a glibc 2.28 floor on the artifacts ([#130](https://github.com/wasabipesto/nice/pull/130)).
- Add opt-in `cubecl-spirv` and `cubecl-metal` build features that route the `cubecl` backend through CubeCL's direct SPIR-V/MSL compilers instead of naga. Fix the kernels' atomic buffers being bound as read-only inputs, which the Metal compiler rejects. The wgpu device name reported in benchmarks and telemetry now includes the shader compiler (`wgpu<wgsl>`, `wgpu<spirv>`, `wgpu<msl>`) ([#126](https://github.com/wasabipesto/nice/pull/126)).
- Add a plane-scoped variant of the CubeCL niceonly compaction queue: each subgroup owns a private queue and drains it without cube-wide barriers. On by default under CubeCL's SPIR-V compiler, where it measures +18-22% on every niceonly scenario on an RX 9070 XT (+11-13% over the previous naga-compiled binary), and under `cubecl-cuda` (+9-18% on an RTX 3090 when the device is the bottleneck); `NICE_CUBECL_PLANE_COMPACT=0|1` overrides. The `gpu` build now includes `cubecl-spirv` so Linux Vulkan users get this path ([#127](https://github.com/wasabipesto/nice/pull/127)).
- The chunk-scan flavor is now chosen per device: CUDA keeps the 64-bit scan, wgpu keeps the 32-bit split scan (measured 1.4-4.6x faster than the 64-bit scan on AMD RDNA4 and Apple M4). `NICE_CUBECL_WIDE=0|1` forces either for A/B runs ([#126](https://github.com/wasabipesto/nice/pull/126)).

## Nice v3.4.1

- Add a new sound cross-end residue filter to the CPU and GPU niceonly paths. The MSD interval-domain analysis now returns a certificate of digits that provably occupy high output positions for every n in a surviving range, and the stride iteration skips any residue whose exact low output digits intersect it ([#122](https://github.com/wasabipesto/nice/pull/122), [#123](https://github.com/wasabipesto/nice/pull/123)).
- Raise the CPU MSD recursion floor from 1000 to 8000: with the cross-end filter absorbing the extra candidates a coarse floor lets through, a two-machine whole-field sweep puts the new optimum at 8000 ([#122](https://github.com/wasabipesto/nice/pull/122)).
- Add an explicit no-MSD bypass to the GPU pipeline. At the maximum GPU recursion floor (now one full processing chunk), chunks are shipped as single descriptors with no endpoint analysis, letting a strong device paired with a weak CPU skip host-side MSD filtering entirely. The adaptive floor controller can now discover this mode on its own ([#123](https://github.com/wasabipesto/nice/pull/123)).
- Compact the cross-end filter's survivors in the CUDA and CubeCL niceonly kernels before checking them. Dispatches whose certificates are all zero (the no-MSD bypass) take the plain path ([#123](https://github.com/wasabipesto/nice/pull/123), [#125](https://github.com/wasabipesto/nice/pull/125)).
- The environment variables `NICE_CUDA_CROSS=0`, `NICE_CUDA_COMPACT=0`, `NICE_CUBECL_CROSS=0`, and `NICE_CUBECL_COMPACT=0` can be set to disable the above behavior for testing and evaluation, but these options will be removed in a future release ([#123](https://github.com/wasabipesto/nice/pull/123)).
- Segment the API `/estimate` endpoint and python fleet estimator by client version to avoid newer/older clients from swaying the EV calculations ([#124](https://github.com/wasabipesto/nice/pull/124)).

## Nice v3.4.0

Client features:

- Add new GPU backends `cubecl`, `cubecl-cuda`, and `vulkan` (experimental) for additional device support in GPU-enabled builds. Notably the client can now utilize consumer AMD GPUs and Apple M-series GPUs. No new configuration is needed, the client will automatically detect and use a compatible backend. To manually select a backend use `--gpu-backend` ([#99](https://github.com/wasabipesto/nice/pull/99)).
- Add WebGPU support to the live WASM web client, which now supports most GPU/OS/browser configurations. (Notable exceptions include AMD cards on linux for most browsers except Firefox Nightly.) ([#105](https://github.com/wasabipesto/nice/pull/105))
- Pipeline the web client similarly to the native client with a small claim buffer and parallel network calls to keep the backend working at full duty cycle. Add a greedy work queue so an efficiency core does not slow down the whole client ([#112](https://github.com/wasabipesto/nice/pull/112), [#116](https://github.com/wasabipesto/nice/pull/116)).
- Add the fleet controller (`fleet/`): a cron-driven explore/exploit loop over the Vast.ai interruptible market utilizing croudsourced benchmarks ([#94](https://github.com/wasabipesto/nice/pull/94), [#108](https://github.com/wasabipesto/nice/pull/108))
- Rework `--benchmark` into a structured sweep over many bases and windows to fill an adjustable time budget (`--benchmark-secs`, default 10) and output (table by default or json with `--benchmark-json`). Reports hardware info, recorded performance, API latency, scheduler correlation IDs, and a synthetic score. Allows uploading to the coordination server for aggregation (manually or with `--benchmark-upload`) ([#90](https://github.com/wasabipesto/nice/pull/90), [#91](https://github.com/wasabipesto/nice/pull/91), [#119](https://github.com/wasabipesto/nice/pull/119)).
- Add the reworked benchmark (and synthetic NiceMark score) process to the web client with similar options for work sizing and uploading the score. This client doesn't have access to the CPU details so can't be used for the fleet estimator but can still be used for bragging rights ([#113](https://github.com/wasabipesto/nice/pull/113)).
- Add opt-in submission telemetry: `--telemetry` attaches hardware, scheduler environment, client config, and client-side processing time to each submission ([#91](https://github.com/wasabipesto/nice/pull/91))
- Client sub-options now imply their umbrella flag: any explicit `--gpu-*` option enables `--gpu`, and any `--benchmark-*` option enables `--benchmark`. Values set through the `NICE_*` environment variables trigger this also, with boolean `NICE_*` environment variables parsing falsy values (`NICE_GPU=false` or `=0` do not enable the GPU) ([#119](https://github.com/wasabipesto/nice/pull/119)).

Performance:

- Replace the MSD filter's common-prefix duplicate/overlap checks with an interval digit-domain analysis (Hall's theorem). Each near-fixed output position of n² and n³ gets a conservative digit domain, and a range is skipped when the constrained positions cannot all receive distinct digits ([#89](https://github.com/wasabipesto/nice/pull/89)).
- Deepen the stride table's LSD filter from k=2 to k=3, removing 15-22% of candidates before the nice check, which was made effective by precomputing specific steps of the process ([#88](https://github.com/wasabipesto/nice/pull/88))
- Raise the CPU MSD recursion floor from 250 to 1000 due to the increased speed of per-digit checks and lower need for MSD prefiltering ([#88](https://github.com/wasabipesto/nice/pull/88))

Website:

- Start the index page data fetches while the plot library is still downloading, draw each chart as its own data arrives instead of waiting for all five requests, and load d3/Plot as vendored files ([#106](https://github.com/wasabipesto/nice/pull/106))
- Draw the notable numbers chart from a subset stored in `cache_notable_numbers` instead of every base's full top-10k list ([#107](https://github.com/wasabipesto/nice/pull/107))
- Rework the web runner's statistics into a session view: fields completed, session totals and rates, best find, a rate-per-field strip chart, and all-time per-browser totals persisted in localStorage ([#114](https://github.com/wasabipesto/nice/pull/114))
- Add a uniform-random occupancy baseline to the web runner's digit distribution histogram overlay so a field's shape is legible against the expected values ([#114](https://github.com/wasabipesto/nice/pull/114))

Backend:

- Add `POST /benchmark`: saves community benchmarks from user reports, validating schema and capping report size ([#91](https://github.com/wasabipesto/nice/pull/91))
- Add `POST /estimate`: predicts per-scenario and blended performance for a hardware configuration from recent benchmark uploads via hierarchical matching (experimental, this endpoint may be removed at any point) ([#93](https://github.com/wasabipesto/nice/pull/93))
- Add `GET /ping`: a static response for client-side latency measurement ([#90](https://github.com/wasabipesto/nice/pull/90))
- Make the scheduled jobs incremental and stream the downsampling aggregation to decrease job time and memory usage. A `--full` option restores the previous sweep-everything behavior for use after manual disqualifications ([#104](https://github.com/wasabipesto/nice/pull/104)).
- Serve `/estimate` from an in-memory benchmark corpus cache instead of re-fetching and re-decoding up to 2000 jsonb reports per request ([#97](https://github.com/wasabipesto/nice/pull/97))
- Optimize claim API calls in the server:
  - Scope the detailed `Thin` claim to the proper frontier chunk by checking for claimable fields ([#100](https://github.com/wasabipesto/nice/pull/100))
  - Scope the detailed `Next` claim to the proper frontier chunk via `chunks.minimum_cl` ([#102](https://github.com/wasabipesto/nice/pull/102))
  - Make queue refills single-flight and run them on the requesting handler's database connection ([#103](https://github.com/wasabipesto/nice/pull/103))
  - Thanks to [Janzert](https://github.com/Janzert) for the report and diagnoses
- Add `common/tests/claim_queries.rs`: the claim path is hand-written SQL with positional bindings that the compiler cannot check, so it is now exercised against a real PostgreSQL. Skipped unless `NICE_TEST_DATABASE_URL` is set ([#100](https://github.com/wasabipesto/nice/pull/100), [#102](https://github.com/wasabipesto/nice/pull/102)).

Maintenance:

- Update all dependencies to their latest versions, including major bumps of malachite, reqwest, sysinfo, and rocket_prometheus. This resolves every RustSec advisory fixable within our dependency tree; the one remaining advisory (RUSTSEC-2026-0258, h2 0.3.x) comes via rocket 0.5's hyper 0.14 and has no fixed 0.3.x release ([#117](https://github.com/wasabipesto/nice/pull/117))
- With reqwest 0.13, default (rustls) builds of the client now verify TLS against the platform trust store instead of bundled webpki roots, and the rustls crypto provider is aws-lc-rs instead of ring. Environments without system CA certificates (bare containers running the static musl binary) now need `ca-certificates` installed; the published docker images already include it ([#117](https://github.com/wasabipesto/nice/pull/117)).
- Enforce formatting and clippy (all targets, including the GPU feature builds) in CI via a new `just lint` recipe, and fix everything it flagged ([#118](https://github.com/wasabipesto/nice/pull/118)).
- Remove the unused `docker/` build-container Dockerfiles, superseded by the cross-compiled release workflow ([#118](https://github.com/wasabipesto/nice/pull/118))

## Nice v3.3.0

- Fix an off-by-one error in get_base_range_natural which dropped one trailing candidate per base where base % 5 ∈ {2,3,4}. Thanks to [Janzert](https://github.com/Janzert) for reporting this!
- Adapt CPU client processing chunk size from field size to reduce memory usage with larger fields and make TQDM output more readable
- Buffer client claims and overlap the requests that refill them. Thanks to [Janzert](https://github.com/Janzert) for implementing this! ([#83](https://github.com/wasabipesto/nice/pull/83))
- Refactor the client to utilize the async API and simplify some codepaths ([#84](https://github.com/wasabipesto/nice/pull/84))
- Remove an unsound MSD x LSD cross-check, which could silently skip valid nice numbers searched in niceonly mode by the CPU client. Submissions from clients 3.2.12-3.1.15 may be annulled while we re-check results. Detailed mode was unaffected ([#87](https://github.com/wasabipesto/nice/pull/87)).

## Nice v3.2.15

- Fix a bug in the nice-only GPU mode where valid nice numbers would be silently skipped in some bases between 10-25 due to an invalid prefilter configuration. This had no effect on live fields but affected some benchmarks.
- Optimize the nice-only GPU path slightly by disabling the prefilter on bases 41 or higher where warp stats indicate it already runs the full check regardless
- Optimize the nice-only GPU path slightly by computing the square's digits and finding conflicts before calculating the cube
- Optimize the nice-only GPU path slightly by accumulating duplicate flags across a chunk and testing once per chunk instead of branching per digit

## Nice v3.2.14

- Rewrite the CUDA GPU path from scratch. Kernels are now JIT-specialized per base and can be verified with libnvrtc. For detailed search, the GPU handles all processing. Not nice-only search, the CPU handles MSD filtering and the GPU handles bulk nice checks. The handoff between the CPU and GPU is adapted over time but can be overridden with `NICE_GPU_MSD_FLOOR` ([#81](https://github.com/wasabipesto/nice/pull/81)).
- Add a field queue for the detailed thin strategy, which speeds up the `/detailed/claim` endpoint about 80% of the time

## Nice v3.2.13
- Add additional fields to the database in bases 52, 53, and 54
- Remove dependence on the DEFAULT_FIELD_SIZE constant. The newest fields are larger than this so the server and client are now able to handle them properly. Detailed search claims will be limited to the previous max size (1e9).
- Add recent search progress and overall leaderboard charts to the site ([#79](https://github.com/wasabipesto/nice/pull/79))
- Add fixed-width compile-time generation to replace div_asign_rem in most cases ([#80](https://github.com/wasabipesto/nice/pull/80))

## Nice v3.2.12

- Asynchronously submit the previous search field and get the next one while processing to reduce network overhead ([#71](https://github.com/wasabipesto/nice/pull/71))
- Reuse server connection between claim/submit requests to reduce network overhead
- Implement higher k-values for the experimental LSD filter, fix a bug and add tests ([#73](https://github.com/wasabipesto/nice/pull/73))
- Implement checks for overlap between the most- and least-significant digits ([#73](https://github.com/wasabipesto/nice/pull/73))
- Implement a CRT stride filter that allows the process to jump between valid candidates instead of iterating over all of them ([#73](https://github.com/wasabipesto/nice/pull/73))
- Configure additional trace logging in the library ([#71](https://github.com/wasabipesto/nice/pull/71))
- Replace `--verbose` and `--quiet` with `--log-level`/`-l` and `--no-progress`/`-n` ([#71](https://github.com/wasabipesto/nice/pull/71))
- Add support for customizing the number of API retries with `--api-max-retries` ([#71](https://github.com/wasabipesto/nice/pull/71))
- Update base bounds script
- Update filter effectiveness script

## Nice v3.2.11

- Implement some coarse but massive optimizations to nice-only processing based on patterns in the most and least significant digits of each range. The exact amount varies by search range but in the current area of interest it is about 2.5x as fast ([#67](https://github.com/wasabipesto/nice/pull/67), [#70](https://github.com/wasabipesto/nice/pull/70)).
- Implement an in-memory queue for nice-only claims so the server can keep up with the increased processing speed. This takes nice-only claim endpoint times from 90-100ms to 3-5ms ([#68](https://github.com/wasabipesto/nice/pull/68)).
- Add a Prometheus exporter to monitor the response times on each API endpoint ([#69](https://github.com/wasabipesto/nice/pull/69))
- Fix an issue where the largest numbers (instead of nicest) are preserved during downsampling
- Decouple detailed chunk processing size from rayon chunk processing size ([#70](https://github.com/wasabipesto/nice/pull/70))
- Start using proper rust logging systems such as env_logger and test-log ([#67](https://github.com/wasabipesto/nice/pull/67), [#70](https://github.com/wasabipesto/nice/pull/70))

## Nice v3.2.10

- Add validate function to client and server to confirm that new results are consistent with past submissions ([#60](https://github.com/wasabipesto/nice/pull/60))
- Add experimental support for GPU acceleration with CUDA ([#59](https://github.com/wasabipesto/nice/pull/59))

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
- Update dependencies ([#44](https://github.com/wasabipesto/nice/pull/44), [#45](https://github.com/wasabipesto/nice/pull/45), [#46](https://github.com/wasabipesto/nice/pull/46), [#47](https://github.com/wasabipesto/nice/pull/47), [#48](https://github.com/wasabipesto/nice/pull/48))

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

- Update dependencies ([#38](https://github.com/wasabipesto/nice/pull/38), [#39](https://github.com/wasabipesto/nice/pull/39), [#42](https://github.com/wasabipesto/nice/pull/42), [#43](https://github.com/wasabipesto/nice/pull/43))
- Bump rust edition to 2024
- Migrate openssl-tls to rustls-tls
- Re-enable multi-platform builds and testing CI
- Allow setting client options via environment variables
- Add client dockerfile and publish images to GHCR

## Nice v3.2.1

- Fixes a bug where the native client would crash upon beginning the second iteration of a --repeat loop due to rayon's thread pool already being initialized.

## Nice v3.2.0

- WebAssembly module, integrated into the search page found at https://nicenumbers.net/search/ ([#29](https://github.com/wasabipesto/nice/pull/29))
- Native client progress bars and parallelization. Both are enabled by default, but you can silence the progress bar with --quiet and you can customize the number of threads with --threads.
