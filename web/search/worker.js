// Web Worker for WASM Nice Number Processing
// This worker runs the computation off the main thread to prevent UI blocking

let wasm = null;
let isInitialized = false;
let shouldStop = false;

// The wasm side serialises near-miss numbers as exact u128 values, but
// JSON.parse turns every number into a double, and anything above 2^53 is
// rounded to the nearest representable one. Base-49 fields sit near 2e16, so
// the near miss 20363742218601559 reached the server as ...560 and the
// submission was rejected: "Unique count for 20363742218601560 is incorrect
// (submitted as 45, server calculated 29)" — the count was right, the number
// had been rounded out from under it.
//
// So the near-miss numbers are quoted before parsing and carried through this
// worker as strings, which is enough for display and accumulation, and
// unquoted again on the way into a submission. Only `number` needs this: the
// distribution's counts are bounded by the field size.
function parseFieldResults(json) {
    return JSON.parse(json.replace(/("number":)(\d+)/g, '$1"$2"'));
}

// Inverse of parseFieldResults' quoting. JSON.stringify would emit the exact
// digits, but as a quoted string; the server wants an integer there.
function stringifySubmission(payload) {
    return JSON.stringify(payload).replace(/"number":"(\d+)"/g, '"number":$1');
}

// Initialize WASM module in worker context.
//
// `sharedModule` is a WebAssembly.Module the page compiled once and
// structured-cloned to every worker. Without it each worker fetches and
// compiles the same ~1.9 MB binary independently, which on a cold cache is
// nine downloads and nine compiles of identical bytes. wasm-bindgen's init
// takes a Module directly and goes straight to WebAssembly.instantiate;
// passing undefined keeps the old fetch-by-URL behaviour.
async function initWasm(sharedModule) {
    try {
        // Import the WASM module
        const wasmModule = await import("./pkg/nice_wasm_client.js");
        await wasmModule.default(sharedModule);
        wasm = wasmModule;
        isInitialized = true;

        // Send initialization success message
        self.postMessage({
            type: "initialized",
            success: true,
        });
    } catch (error) {
        self.postMessage({
            type: "initialized",
            success: false,
            error: error.message,
        });
    }
}

// Process numbers using WASM chunk processing
function processDetailedWithProgress(claimDataJson, username) {
    const claimData = JSON.parse(claimDataJson);
    const base = claimData.base;
    const rangeStart = BigInt(claimData.range_start);
    const rangeEnd = BigInt(claimData.range_end);
    const rangeSize = rangeEnd - rangeStart;

    // Send initial status
    self.postMessage({
        type: "progress",
        percent: 0,
        message: "Starting detailed processing...",
        processedCount: 0,
        uniqueDistribution: new Map(),
        niceNumbers: [],
    });

    const allNiceNumbers = [];
    const uniqueDistribution = new Map();

    // Initialize distribution map
    for (let i = 1; i <= base; i++) {
        uniqueDistribution.set(i, 0);
    }

    let processed = BigInt(0);
    const chunkSize = BigInt(100000); // Large chunks for WASM efficiency
    let lastProgressUpdate = Date.now();
    const progressUpdateInterval = 1000; // Update every 1 second

    for (
        let current = rangeStart;
        current < rangeEnd && !shouldStop;
        current += chunkSize
    ) {
        const chunkEnd =
            current + chunkSize > rangeEnd ? rangeEnd : current + chunkSize;

        // Process entire chunk in WASM
        const chunkResultJson = wasm.process_chunk_wasm(
            current.toString(),
            chunkEnd.toString(),
            base,
        );

        const chunkResult = parseFieldResults(chunkResultJson);

        // Merge nice numbers
        allNiceNumbers.push(...chunkResult.nice_numbers);

        // Update distribution ("distribution" in FieldResults; the old
        // "distribution_updates" name predates a serialization rename)
        for (const entry of chunkResult.distribution) {
            const currentCount = uniqueDistribution.get(entry.num_uniques) || 0;
            uniqueDistribution.set(
                entry.num_uniques,
                currentCount + entry.count,
            );
        }

        processed += chunkSize;

        // Send progress update
        const now = Date.now();
        if (now - lastProgressUpdate > progressUpdateInterval) {
            const percent = Number((processed * BigInt(100)) / rangeSize);
            const processedCount = Number(processed);
            const totalCount = Number(rangeSize);

            self.postMessage({
                type: "progress",
                percent: percent,
                message: `Processed ${processedCount.toLocaleString()} / ${totalCount.toLocaleString()} numbers`,
                processedCount: processedCount,
                uniqueDistribution: uniqueDistribution,
                niceNumbers: allNiceNumbers,
            });

            lastProgressUpdate = now;
        }
    }

    if (shouldStop) {
        self.postMessage({
            type: "stopped",
            message: "Processing stopped by user",
        });
        return;
    }

    // Convert results back to server format
    const serverNiceNumbers = allNiceNumbers.map((nn) => ({
        number: nn.number,
        num_uniques: nn.num_uniques,
    }));

    const serverDistribution = Array.from(uniqueDistribution.entries())
        .map(([num_uniques, count]) => ({
            num_uniques: num_uniques,
            count: count,
        }))
        .sort((a, b) => a.num_uniques - b.num_uniques);

    const result = {
        claim_id: claimData.claim_id,
        username: username,
        client_version: "3.0.0-wasm-worker",
        unique_distribution: serverDistribution,
        nice_numbers: serverNiceNumbers,
    };

    return stringifySubmission(result);
}

// Process numbers on the GPU (WebGPU via CubeCL) with progress updates.
// Slices the claim so progress ticks and stop signals stay responsive; each
// slice is one async wasm call that runs entirely on the GPU.
async function processDetailedGpu(claimDataJson, username) {
    const claimData = JSON.parse(claimDataJson);
    const base = claimData.base;
    const rangeStart = BigInt(claimData.range_start);
    const rangeEnd = BigInt(claimData.range_end);
    const rangeSize = rangeEnd - rangeStart;

    self.postMessage({
        type: "progress",
        percent: 0,
        message: "Starting GPU processing...",
        processedCount: 0,
        uniqueDistribution: new Map(),
        niceNumbers: [],
    });

    const allNiceNumbers = [];
    const uniqueDistribution = new Map();
    for (let i = 1; i <= base; i++) {
        uniqueDistribution.set(i, 0);
    }

    // Every slice ends in a GPU->CPU readback, which drains the queue and
    // holds the device idle until the CPU has the data. Sizing a slice to one
    // of the kernel's internal batches therefore alternates a single dispatch
    // with a full stall, which is most of where the browser client's time went
    // — the card measured 18-20% busy. Several batches per slice let the
    // dispatches queue back to back so only the last is waited on.
    //
    // The batch size comes from the wasm build rather than being repeated
    // here, so the two cannot drift apart.
    let batchSize = BigInt(32000000);
    try {
        const info = JSON.parse(wasm.gpu_build_info());
        if (info && info.batch_size) {
            batchSize = BigInt(info.batch_size);
        }
    } catch {
        /* keep the default above */
    }
    const megaBig = BigInt(1000000);
    // Eight progress ticks a field, but never a slice so small that the GPU
    // spends its time waiting rather than working.
    let sliceSize = rangeSize / BigInt(8);
    const minSlice = batchSize * BigInt(8);
    if (sliceSize < minSlice) {
        sliceSize = minSlice;
    }
    if (sliceSize > megaBig) {
        sliceSize = (sliceSize / megaBig) * megaBig;
    }
    if (sliceSize > rangeSize || sliceSize < BigInt(1)) {
        sliceSize = rangeSize;
    }

    let processed = BigInt(0);
    for (
        let current = rangeStart;
        current < rangeEnd && !shouldStop;
        current += sliceSize
    ) {
        const sliceEnd =
            current + sliceSize > rangeEnd ? rangeEnd : current + sliceSize;

        const sliceResultJson = await wasm.process_chunk_gpu(
            current.toString(),
            sliceEnd.toString(),
            base,
        );
        const sliceResult = parseFieldResults(sliceResultJson);

        allNiceNumbers.push(...sliceResult.nice_numbers);
        for (const entry of sliceResult.distribution) {
            const currentCount = uniqueDistribution.get(entry.num_uniques) || 0;
            uniqueDistribution.set(
                entry.num_uniques,
                currentCount + entry.count,
            );
        }

        processed += sliceEnd - current;
        const percent = Number((processed * BigInt(100)) / rangeSize);
        self.postMessage({
            type: "progress",
            percent: percent,
            message: `GPU processed ${Number(processed).toLocaleString()} / ${Number(rangeSize).toLocaleString()} numbers`,
            processedCount: Number(processed),
            uniqueDistribution: uniqueDistribution,
            niceNumbers: allNiceNumbers,
        });
    }

    if (shouldStop) {
        self.postMessage({
            type: "stopped",
            message: "Processing stopped by user",
        });
        return null;
    }

    const serverNiceNumbers = allNiceNumbers.map((nn) => ({
        number: nn.number,
        num_uniques: nn.num_uniques,
    }));
    const serverDistribution = Array.from(uniqueDistribution.entries())
        .map(([num_uniques, count]) => ({
            num_uniques: num_uniques,
            count: count,
        }))
        .sort((a, b) => a.num_uniques - b.num_uniques);

    return stringifySubmission({
        claim_id: claimData.claim_id,
        username: username,
        client_version: "3.0.0-wasm-webgpu",
        unique_distribution: serverDistribution,
        nice_numbers: serverNiceNumbers,
    });
}

// Handle messages from main thread
self.onmessage = async function (e) {
    const { type, data } = e.data;

    switch (type) {
        case "init":
            await initWasm(data?.module);
            break;

        case "process":
            if (!isInitialized) {
                self.postMessage({
                    type: "error",
                    error: "WASM not initialized",
                });
                return;
            }

            shouldStop = false;
            const { claimData, username } = data;

            try {
                const startTime = Date.now();
                const resultJson = processDetailedWithProgress(
                    JSON.stringify(claimData),
                    username,
                );

                if (!shouldStop && resultJson) {
                    const endTime = Date.now();
                    const elapsedSeconds = (endTime - startTime) / 1000;

                    self.postMessage({
                        type: "complete",
                        result: parseFieldResults(resultJson),
                        elapsedSeconds: elapsedSeconds,
                    });
                }
            } catch (error) {
                self.postMessage({
                    type: "error",
                    error: error.message,
                });
            }
            break;

        case "init_gpu":
            if (!isInitialized) {
                self.postMessage({
                    type: "gpu_initialized",
                    success: false,
                    error: "WASM not initialized",
                });
                return;
            }
            try {
                if (wasm.gpu_build_info) {
                    self.postMessage({
                        type: "gpu_build_info",
                        info: wasm.gpu_build_info(),
                    });
                }
                const adapterName = await wasm.gpu_init();
                self.postMessage({
                    type: "gpu_initialized",
                    success: true,
                    adapterName: adapterName,
                });
            } catch (error) {
                self.postMessage({
                    type: "gpu_initialized",
                    success: false,
                    error: String(error),
                });
            }
            break;

        case "process_gpu":
            if (!isInitialized) {
                self.postMessage({
                    type: "error",
                    error: "WASM not initialized",
                });
                return;
            }
            shouldStop = false;
            try {
                const gpuStart = Date.now();
                const gpuResultJson = await processDetailedGpu(
                    JSON.stringify(e.data.data.claimData),
                    e.data.data.username,
                );
                if (!shouldStop && gpuResultJson) {
                    self.postMessage({
                        type: "complete",
                        result: parseFieldResults(gpuResultJson),
                        elapsedSeconds: (Date.now() - gpuStart) / 1000,
                    });
                }
            } catch (error) {
                self.postMessage({
                    type: "error",
                    error: String(error),
                });
            }
            break;

        case "stop":
            shouldStop = true;
            self.postMessage({
                type: "stopped",
                message: "Stop signal received",
            });
            break;

        case "benchmark":
            // Return benchmark data
            const benchmarkData = {
                // A number, matching what the server sends and what /submit
                // expects back — the benchmark never submits, but the shape
                // should not differ from a real field's.
                claim_id: 0,
                base: 40,
                range_start: "1916284264916",
                range_end: "1916294264916",
                range_size: "10000000",
            };

            self.postMessage({
                type: "benchmark_data",
                data: benchmarkData,
            });
            break;

        default:
            console.warn("Unknown message type:", type);
    }
};

// Handle errors
self.onerror = function (error) {
    self.postMessage({
        type: "error",
        error: `Worker error: ${error.message}`,
    });
};
