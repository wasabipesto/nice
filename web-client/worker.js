// Web Worker for WASM Nice Number Processing
// This worker runs the computation off the main thread to prevent UI blocking

let wasm = null;
let isInitialized = false;
let shouldStop = false;

// Initialize WASM module in worker context
async function initWasm() {
    try {
        // Import the WASM module
        const wasmModule = await import("./pkg/nice_web_client.js");
        await wasmModule.default();
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
    });

    const allNiceNumbers = [];
    const uniqueDistribution = new Map();

    // Initialize distribution map
    for (let i = 1; i <= base; i++) {
        uniqueDistribution.set(i, 0);
    }

    let processed = BigInt(0);
    const chunkSize = BigInt(10000); // Large chunks for WASM efficiency
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

        const chunkResult = JSON.parse(chunkResultJson);

        // Merge nice numbers
        allNiceNumbers.push(...chunkResult.nice_numbers);

        // Update distribution
        for (const entry of chunkResult.distribution_updates) {
            const currentCount = uniqueDistribution.get(entry.num_uniques) || 0;
            uniqueDistribution.set(
                entry.num_uniques,
                currentCount + entry.count,
            );
        }

        processed += BigInt(chunkResult.processed_count);

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
        number: parseInt(nn.number),
        num_uniques: nn.num_uniques,
    }));

    const serverDistribution = Array.from(uniqueDistribution.entries())
        .map(([num_uniques, count]) => ({
            num_uniques: num_uniques,
            count: count,
        }))
        .sort((a, b) => a.num_uniques - b.num_uniques);

    const result = {
        claim_id: parseInt(claimData.claim_id),
        username: username,
        client_version: "3.0.0-wasm-worker",
        unique_distribution: serverDistribution,
        nice_numbers: serverNiceNumbers,
    };

    return JSON.stringify(result);
}

// Handle messages from main thread
self.onmessage = async function (e) {
    const { type, data } = e.data;

    switch (type) {
        case "init":
            await initWasm();
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
                        result: JSON.parse(resultJson),
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
                claim_id: "0",
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
