// Multi-threaded Worker Pool Manager for WASM Nice Number Processing
// This manages multiple workers for parallel processing

// Compile the wasm binary once for the whole page and hand the resulting
// WebAssembly.Module to every worker. Without this each worker fetches and
// compiles its own copy of the same ~1.9 MB file: on a cold cache that is one
// download and one compile per thread, of identical bytes. The promise is
// memoised, so callers racing at startup share a single fetch.
//
// A failure here is not fatal — resolving to undefined puts each worker back
// on its own fetch-by-URL path.
function niceWasmModule() {
    if (!globalThis.__niceWasmModulePromise) {
        globalThis.__niceWasmModulePromise = WebAssembly.compileStreaming(
            fetch("./pkg/nice_wasm_client_bg.wasm"),
        ).catch((e) => {
            console.warn(
                "shared wasm compile failed; each worker will load its own",
                e,
            );
            return undefined;
        });
    }
    return globalThis.__niceWasmModulePromise;
}

class WorkerPool {
    constructor(options = {}) {
        // Default to 80% of available cores
        const availableCores = navigator.hardwareConcurrency || 4;
        const defaultMaxWorkers = Math.max(1, Math.floor(availableCores * 0.8));

        this.maxWorkers = options.maxWorkers || defaultMaxWorkers;
        this.progressUpdateInterval = options.progressUpdateInterval || 500; // 500ms default

        this.workers = [];
        this.isInitialized = false;
        this.activeJobs = new Map();
        this.jobIdCounter = 0;

        // Aggregated results
        this.aggregatedResults = {
            niceNumbers: [],
            uniqueDistribution: new Map(),
            errors: [],
        };

        // Progress tracking
        this.progressCallback = null;
        this.completeCallback = null;
        this.errorCallback = null;
        this.currentJobId = null;
        this.workerProgress = new Map(); // Track individual worker progress
        this.lastProgressUpdate = 0;

        // Current job data
        this.currentClaimId = null;
        this.currentUsername = null;

        // Reported by each worker at init (they read it from the wasm
        // build); stamped onto the aggregated submission.
        this.clientVersion = "unknown";
    }

    async initialize() {
        try {
            const sharedModule = await niceWasmModule();
            // Create worker instances
            for (let i = 0; i < this.maxWorkers; i++) {
                const worker = new Worker("./worker.js");
                const workerInfo = {
                    id: i,
                    worker: worker,
                    isReady: false,
                    currentJob: null,
                };

                // Set up message handler for this worker
                worker.onmessage = (e) =>
                    this.handleWorkerMessage(workerInfo, e);
                worker.onerror = (e) => this.handleWorkerError(workerInfo, e);

                this.workers.push(workerInfo);

                // Initialize this worker
                worker.postMessage({
                    type: "init",
                    data: { module: sharedModule },
                });
            }

            // Wait for all workers to initialize
            await this.waitForInitialization();
            this.isInitialized = true;

            console.log(
                `Worker pool initialized with ${this.maxWorkers} workers`,
            );
            return true;
        } catch (error) {
            console.error("Failed to initialize worker pool:", error);
            throw error;
        }
    }

    waitForInitialization() {
        return new Promise((resolve, reject) => {
            let initializedCount = 0;
            const timeout = setTimeout(() => {
                reject(new Error("Worker initialization timeout"));
            }, 30000); // 30 second timeout

            const checkInitialization = () => {
                if (initializedCount >= this.maxWorkers) {
                    clearTimeout(timeout);
                    resolve();
                }
            };

            this.workers.forEach((workerInfo) => {
                const originalHandler = workerInfo.worker.onmessage;
                workerInfo.worker.onmessage = (e) => {
                    if (e.data.type === "initialized") {
                        if (e.data.success) {
                            workerInfo.isReady = true;
                            initializedCount++;
                            if (e.data.version) {
                                this.clientVersion = e.data.version;
                            }
                            checkInitialization();
                        } else {
                            clearTimeout(timeout);
                            reject(
                                new Error(
                                    `Worker ${workerInfo.id} failed to initialize: ${e.data.error}`,
                                ),
                            );
                        }
                    }
                    // Restore original handler for future messages
                    workerInfo.worker.onmessage = originalHandler;
                    if (originalHandler) originalHandler(e);
                };
            });
        });
    }

    async processClaimData(claimData, username, callbacks = {}) {
        if (!this.isInitialized) {
            throw new Error("Worker pool not initialized");
        }

        if (!claimData || !claimData.range_start || !claimData.range_end) {
            throw new Error("Invalid claim data provided");
        }

        this.currentJobId = ++this.jobIdCounter;
        this.progressCallback = callbacks.onProgress;
        this.completeCallback = callbacks.onComplete;
        this.errorCallback = callbacks.onError;

        // Store claim_id and username for later use
        this.currentClaimId = parseInt(claimData.claim_id) || 0;
        this.currentUsername = username;

        // Reset aggregated results
        this.resetAggregatedResults();

        try {
            const rangeStart = BigInt(claimData.range_start);
            const rangeEnd = BigInt(claimData.range_end);
            const totalRange = rangeEnd - rangeStart;

            if (totalRange <= 0n) {
                throw new Error("Invalid range: start must be less than end");
            }

            // A queue of sub-ranges instead of a static 1/N split. The
            // browser schedules workers on whatever cores it likes —
            // E-cores, throttled cores — and under a static split the
            // slowest worker sets the field's finish time while the rest
            // sit idle. A queue keeps every worker busy until the work
            // itself runs out, and lets a stop land within one sub-range
            // instead of one Nth of the field.
            //
            // ~32 pieces per worker balances the tail against per-message
            // overhead; a piece is never smaller than two of the worker's
            // internal 100k chunks, and never so large that a worker sits
            // idle from the start.
            let subSize = totalRange / (BigInt(this.maxWorkers) * 32n);
            if (subSize < 200000n) subSize = 200000n;
            const evenSplit = totalRange / BigInt(this.maxWorkers);
            if (subSize > evenSplit) subSize = evenSplit;
            if (subSize < 1n) subSize = 1n;

            this.workQueue = [];
            for (let s = rangeStart; s < rangeEnd; s += subSize) {
                const e = s + subSize > rangeEnd ? rangeEnd : s + subSize;
                this.workQueue.push([s, e]);
            }
            this.baseClaimData = claimData;
            this.totalRange = totalRange;
            this.completedNumbers = 0n;
            this.jobStartTime = Date.now();

            console.log(
                `Queued ${this.workQueue.length} sub-ranges of ~${subSize} for ${this.maxWorkers} workers`,
            );

            // Prime every worker; each comes back for more on completion.
            this.workers.forEach((workerInfo) => this.assignNext(workerInfo));
        } catch (error) {
            if (this.errorCallback) {
                this.errorCallback(error.message);
            }
            throw error;
        }
    }

    // Hand a worker the next sub-range, or leave it idle when the queue is
    // dry (the field finishes when the last active worker reports in).
    assignNext(workerInfo) {
        const next = this.workQueue.shift();
        if (!next) {
            workerInfo.currentJob = null;
            this.activeJobs.delete(workerInfo.id);
            return;
        }
        const [subStart, subEnd] = next;
        const job = {
            workerId: workerInfo.id,
            jobId: this.currentJobId,
            size: subEnd - subStart,
        };
        workerInfo.currentJob = job;
        this.activeJobs.set(workerInfo.id, job);
        workerInfo.worker.postMessage({
            type: "process",
            data: {
                claimData: {
                    ...this.baseClaimData,
                    range_start: subStart.toString(),
                    range_end: subEnd.toString(),
                },
                username: this.currentUsername,
            },
        });
    }

    handleWorkerMessage(workerInfo, e) {
        const { type, data, ...rest } = e.data;
        const job = this.activeJobs.get(workerInfo.id);

        if (!job || job.jobId !== this.currentJobId) {
            // Ignore messages from old jobs
            return;
        }

        switch (type) {
            case "progress":
                this.handleProgress(workerInfo, rest);
                break;

            case "complete":
                this.handleComplete(
                    workerInfo,
                    rest.result,
                    rest.elapsedSeconds,
                );
                break;

            case "error":
                this.handleError(workerInfo, rest.error);
                break;

            case "stopped":
                this.handleStopped(workerInfo);
                break;

            default:
                console.warn(
                    `Unknown message type from worker ${workerInfo.id}:`,
                    type,
                );
        }
    }

    handleProgress(workerInfo, progressData) {
        // Store this worker's progress data
        this.workerProgress.set(workerInfo.id, progressData);

        // Throttle progress updates to avoid UI flooding
        const now = Date.now();
        if (now - this.lastProgressUpdate > this.progressUpdateInterval) {
            this.lastProgressUpdate = now;

            if (this.progressCallback) {
                // Calculate overall progress with real data
                const overallProgress = this.calculateOverallProgress();
                this.progressCallback(overallProgress);
            }
        }
    }

    handleComplete(workerInfo, result, elapsedSeconds) {
        console.log(
            `Worker ${workerInfo.id} completed processing. Result:`,
            result,
        );

        // Validate result structure
        if (!result || typeof result !== "object") {
            console.error(
                `Worker ${workerInfo.id} returned invalid result:`,
                result,
            );
            this.handleError(workerInfo, "Invalid result format from worker");
            return;
        }

        // Aggregate this sub-range's results and account for its numbers;
        // the worker's live-progress entry is superseded by the total.
        this.aggregateWorkerResults(result);
        const job = workerInfo.currentJob;
        this.completedNumbers += job ? job.size : 0n;
        this.workerProgress.delete(workerInfo.id);
        this.activeJobs.delete(workerInfo.id);
        workerInfo.currentJob = null;

        // Back to the queue for more; the field is done when the queue is
        // dry and the last active worker has reported in.
        this.assignNext(workerInfo);
        if (this.activeJobs.size === 0 && this.workQueue.length === 0) {
            // Wall time of the whole field, not the last worker's own
            // timer: with a queue the two genuinely differ.
            this.handleAllWorkersComplete(
                (Date.now() - this.jobStartTime) / 1000,
            );
        }
    }

    handleError(workerInfo, error) {
        // A failed sub-range fails the field. The previous version could
        // "complete with partial results" after worker failures, but a
        // partial distribution can never be submitted — the server checks
        // that the counts cover the whole range — so the only honest
        // outcome is an error the page can act on.
        console.error(`Worker ${workerInfo.id} error:`, error);
        this.activeJobs.delete(workerInfo.id);
        workerInfo.currentJob = null;
        this.stopProcessing();
        if (this.errorCallback) {
            this.errorCallback(`Worker ${workerInfo.id}: ${error}`);
        }
    }

    handleStopped(workerInfo) {
        console.log(`Worker ${workerInfo.id} stopped`);
        this.activeJobs.delete(workerInfo.id);
        workerInfo.currentJob = null;
    }

    handleWorkerError(workerInfo, error) {
        console.error(`Worker ${workerInfo.id} script error:`, error);
        this.handleError(workerInfo, `Script error: ${error.message}`);
    }

    calculateOverallProgress() {
        // Aggregate real-time progress from all workers
        let totalProcessed = 0;
        let activeWorkerCount = 0;
        const combinedDistribution = new Map();
        const combinedNiceNumbers = [];

        // Aggregate data from active workers
        this.workerProgress.forEach((progressData, workerId) => {
            if (progressData) {
                totalProcessed += progressData.processedCount || 0;
                activeWorkerCount++;

                // Merge distributions
                if (progressData.uniqueDistribution) {
                    if (progressData.uniqueDistribution instanceof Map) {
                        progressData.uniqueDistribution.forEach(
                            (count, numUniques) => {
                                const currentCount =
                                    combinedDistribution.get(numUniques) || 0;
                                combinedDistribution.set(
                                    numUniques,
                                    currentCount + count,
                                );
                            },
                        );
                    }
                }

                // Merge nice numbers
                if (
                    progressData.niceNumbers &&
                    Array.isArray(progressData.niceNumbers)
                ) {
                    combinedNiceNumbers.push(...progressData.niceNumbers);
                }
            }
        });

        // Add completed workers' results
        combinedNiceNumbers.push(...this.aggregatedResults.niceNumbers);
        this.aggregatedResults.uniqueDistribution.forEach(
            (count, numUniques) => {
                const currentCount = combinedDistribution.get(numUniques) || 0;
                combinedDistribution.set(numUniques, currentCount + count);
            },
        );

        // Progress is real numbers over the real total: completed
        // sub-ranges are accounted exactly, active workers add their
        // in-flight counts.
        const inProgress = totalProcessed;
        const done = Number(this.completedNumbers) + inProgress;
        const overallPercent = Math.min(
            99,
            Math.floor((100 * done) / Number(this.totalRange || 1n)),
        );

        return {
            type: "progress",
            percent: overallPercent,
            message: `Processing with ${this.maxWorkers} workers... Active: ${activeWorkerCount}, Queued: ${this.workQueue?.length ?? 0}`,
            processedCount: done,
            uniqueDistribution: combinedDistribution,
            // Exact u128 digits arrive as strings; subtracting them would
            // go through a double and misorder anything above 2^53.
            niceNumbers: combinedNiceNumbers.sort((a, b) => {
                const x = BigInt(a.number);
                const y = BigInt(b.number);
                return x < y ? -1 : x > y ? 1 : 0;
            }),
        };
    }

    aggregateWorkerResults(result) {
        console.log("🔍 aggregateWorkerResults called with:", result);
        try {
            // Merge nice numbers with validation
            if (result.nice_numbers && Array.isArray(result.nice_numbers)) {
                const validNiceNumbers = result.nice_numbers.filter(
                    (nn) =>
                        nn &&
                        typeof nn.number !== "undefined" &&
                        typeof nn.num_uniques !== "undefined",
                );
                this.aggregatedResults.niceNumbers.push(...validNiceNumbers);
            }

            // Merge unique distribution with validation
            if (
                result.unique_distribution &&
                Array.isArray(result.unique_distribution)
            ) {
                result.unique_distribution.forEach((entry) => {
                    if (
                        entry &&
                        typeof entry.num_uniques === "number" &&
                        typeof entry.count === "number"
                    ) {
                        const currentCount =
                            this.aggregatedResults.uniqueDistribution.get(
                                entry.num_uniques,
                            ) || 0;
                        this.aggregatedResults.uniqueDistribution.set(
                            entry.num_uniques,
                            currentCount + entry.count,
                        );
                    }
                });
            }
        } catch (error) {
            console.error("Error aggregating worker results:", error);
        }
    }

    handleAllWorkersComplete(elapsedSeconds) {
        console.log("All workers completed processing");

        // Sort nice numbers by value. These are strings (exact u128 digits,
        // see parseFieldResults in worker.js), and subtracting them would go
        // through a double and misorder anything above 2^53, so compare as
        // BigInt.
        this.aggregatedResults.niceNumbers.sort((a, b) => {
            const x = BigInt(a.number);
            const y = BigInt(b.number);
            return x < y ? -1 : x > y ? 1 : 0;
        });

        // Convert distribution map to server format
        const serverDistribution = Array.from(
            this.aggregatedResults.uniqueDistribution.entries(),
        )
            .map(([num_uniques, count]) => ({
                num_uniques: num_uniques,
                count: count,
            }))
            .sort((a, b) => a.num_uniques - b.num_uniques);

        // Use stored claim_id and username from processing initialization
        const claim_id = this.currentClaimId || 0;
        const username = this.currentUsername || "anonymous";

        const finalResult = {
            claim_id: claim_id,
            username: username,
            client_version: `${this.clientVersion}-wasm-worker`,
            unique_distribution: serverDistribution,
            nice_numbers: this.aggregatedResults.niceNumbers,
        };

        console.log("Final result being sent:", finalResult);

        if (this.completeCallback) {
            this.completeCallback({
                type: "complete",
                result: finalResult,
                elapsedSeconds: elapsedSeconds,
            });
        }

        // Clear claim data after using it
        this.currentClaimId = null;
        this.currentUsername = null;
    }

    resetAggregatedResults() {
        this.aggregatedResults = {
            niceNumbers: [],
            uniqueDistribution: new Map(),
            errors: [],
        };
        this.workerProgress.clear();
        this.lastProgressUpdate = 0;
        this.workQueue = [];
        this.completedNumbers = 0n;
        this.totalRange = 0n;
    }

    stopProcessing() {
        console.log("Stopping all workers...");

        // Send stop signal to all workers
        this.workers.forEach((workerInfo) => {
            try {
                workerInfo.worker.postMessage({ type: "stop" });
            } catch (error) {
                console.warn(`Failed to stop worker ${workerInfo.id}:`, error);
            }
        });

        // Clear active jobs and the remaining work; workers finish only
        // their current sub-range.
        this.workQueue = [];
        this.activeJobs.clear();
        this.currentJobId = null;

        // Clear worker jobs
        this.workers.forEach((workerInfo) => {
            workerInfo.currentJob = null;
        });

        // Reset aggregated results
        this.resetAggregatedResults();
    }

    terminate() {
        console.log("Terminating worker pool...");

        this.workers.forEach((workerInfo) => {
            workerInfo.worker.terminate();
        });

        this.workers = [];
        this.isInitialized = false;
        this.activeJobs.clear();
        this.currentClaimId = null;
        this.currentUsername = null;
    }

    // One-shot request/response against the first worker: send `message`,
    // resolve with the first reply of `responseType`, pass everything else
    // to the normal handler. The wasm instances live in the workers, so this
    // is how the page asks the wasm build a question.
    requestFromWorker(message, responseType, timeoutMs = 5000) {
        if (!this.isInitialized || this.workers.length === 0) {
            throw new Error("Worker pool not initialized");
        }

        return new Promise((resolve, reject) => {
            const worker = this.workers[0].worker;
            const timeout = setTimeout(() => {
                worker.onmessage = originalHandler;
                reject(new Error(`${responseType} timeout`));
            }, timeoutMs);

            const originalHandler = worker.onmessage;
            worker.onmessage = (e) => {
                if (e.data.type === responseType) {
                    clearTimeout(timeout);
                    worker.onmessage = originalHandler;
                    resolve(e.data);
                } else if (originalHandler) {
                    originalHandler(e);
                }
            };

            worker.postMessage(message);
        });
    }

    async getBenchmarkData() {
        const reply = await this.requestFromWorker(
            { type: "benchmark" },
            "benchmark_data",
        );
        return reply.data;
    }

    // The shared benchmark scenario plan from the wasm build, or null on a
    // build that predates it.
    async getBenchmarkPlan() {
        const reply = await this.requestFromWorker(
            { type: "benchmark_plan" },
            "benchmark_plan",
        );
        return reply.plan;
    }

    // NiceMark for measured rates, scored inside the wasm build against the
    // same references the native client uses.
    async getNicemarkScore(rates, gpu) {
        const reply = await this.requestFromWorker(
            { type: "nicemark", data: { rates: rates, gpu: gpu } },
            "nicemark",
        );
        return reply.score;
    }

    getWorkerCount() {
        return this.maxWorkers;
    }

    isReady() {
        return this.isInitialized && this.workers.every((w) => w.isReady);
    }
}

// Export for use in main thread
window.WorkerPool = WorkerPool;
