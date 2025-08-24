// Multi-threaded Worker Pool Manager for WASM Nice Number Processing
// This manages multiple workers for parallel processing

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
            totalProcessed: 0,
            completedWorkers: 0,
            errors: [],
        };

        // Progress tracking
        this.progressCallback = null;
        this.completeCallback = null;
        this.errorCallback = null;
        this.currentJobId = null;
        this.workerProgress = new Map(); // Track individual worker progress
        this.lastProgressUpdate = 0;
    }

    async initialize() {
        try {
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
                worker.postMessage({ type: "init" });
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

        // Reset aggregated results
        this.resetAggregatedResults();

        try {
            // Calculate range division
            const rangeStart = BigInt(claimData.range_start);
            const rangeEnd = BigInt(claimData.range_end);
            const totalRange = rangeEnd - rangeStart;

            if (totalRange <= 0n) {
                throw new Error("Invalid range: start must be less than end");
            }
            const subRangeSize = totalRange / BigInt(this.maxWorkers);

            // Handle edge case where range is smaller than worker count
            const effectiveWorkers =
                totalRange < BigInt(this.maxWorkers)
                    ? Number(totalRange)
                    : this.maxWorkers;

            console.log(
                `Dividing range ${totalRange} among ${effectiveWorkers} workers (${subRangeSize} per worker)`,
            );

            // Create sub-jobs for each worker
            const jobs = [];
            for (let i = 0; i < effectiveWorkers; i++) {
                const subRangeStart = rangeStart + BigInt(i) * subRangeSize;
                const subRangeEnd =
                    i === effectiveWorkers - 1
                        ? rangeEnd
                        : subRangeStart + subRangeSize;

                const subClaimData = {
                    ...claimData,
                    range_start: subRangeStart.toString(),
                    range_end: subRangeEnd.toString(),
                };

                jobs.push({
                    workerId: i,
                    claimData: subClaimData,
                    username: username,
                    jobId: this.currentJobId,
                });
            }

            // Start processing on all workers
            jobs.forEach((job) => {
                const workerInfo = this.workers[job.workerId];
                workerInfo.currentJob = job;
                this.activeJobs.set(job.workerId, job);

                workerInfo.worker.postMessage({
                    type: "process",
                    data: {
                        claimData: job.claimData,
                        username: job.username,
                    },
                });
            });

            console.log(
                `Started processing with ${this.maxWorkers} workers, range divided into sub-ranges`,
            );
        } catch (error) {
            if (this.errorCallback) {
                this.errorCallback(error.message);
            }
            throw error;
        }
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
        console.log(`Worker ${workerInfo.id} completed processing`);

        // Validate result structure
        if (!result || typeof result !== "object") {
            console.error(
                `Worker ${workerInfo.id} returned invalid result:`,
                result,
            );
            this.handleError(workerInfo, "Invalid result format from worker");
            return;
        }

        // Aggregate this worker's results
        this.aggregateWorkerResults(result);
        this.aggregatedResults.completedWorkers++;

        // Clean up this job
        this.activeJobs.delete(workerInfo.id);
        workerInfo.currentJob = null;

        // Check if all workers are done (use active jobs count for accuracy)
        const expectedWorkers = Math.min(
            this.maxWorkers,
            this.activeJobs.size + this.aggregatedResults.completedWorkers,
        );
        if (
            this.aggregatedResults.completedWorkers >= expectedWorkers ||
            this.activeJobs.size === 0
        ) {
            this.handleAllWorkersComplete(elapsedSeconds);
        }
    }

    handleError(workerInfo, error) {
        console.error(`Worker ${workerInfo.id} error:`, error);
        this.aggregatedResults.errors.push({
            workerId: workerInfo.id,
            error: error,
            timestamp: new Date().toISOString(),
        });

        // Clean up failed worker job
        this.activeJobs.delete(workerInfo.id);
        if (workerInfo.currentJob) {
            workerInfo.currentJob = null;
        }

        // If too many workers fail, abort the entire operation
        const failureThreshold = Math.ceil(this.maxWorkers / 2); // Allow up to half to fail
        if (this.aggregatedResults.errors.length >= failureThreshold) {
            this.stopProcessing();
            if (this.errorCallback) {
                this.errorCallback(
                    `Too many worker failures (${this.aggregatedResults.errors.length}/${this.maxWorkers}). Aborting operation.`,
                );
            }
            return;
        }

        if (this.errorCallback) {
            this.errorCallback(`Worker ${workerInfo.id}: ${error}`);
        }

        // Check if we should complete with partial results
        if (
            this.activeJobs.size === 0 &&
            this.aggregatedResults.completedWorkers > 0
        ) {
            console.warn(
                `Completing with partial results due to worker failures`,
            );
            this.handleAllWorkersComplete(0);
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
        let totalPercent = 0;
        let totalProcessed = 0;
        let activeWorkerCount = 0;
        const combinedDistribution = new Map();
        const combinedNiceNumbers = [];

        // Aggregate data from active workers
        this.workerProgress.forEach((progressData, workerId) => {
            if (progressData) {
                totalPercent += progressData.percent || 0;
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

        // Calculate average progress
        const avgPercent =
            activeWorkerCount > 0 ? totalPercent / activeWorkerCount : 0;
        const completedRatio =
            this.aggregatedResults.completedWorkers / this.maxWorkers;
        const overallPercent = Math.min(
            99,
            Math.floor(
                avgPercent * (1 - completedRatio) + completedRatio * 100,
            ),
        );

        return {
            type: "progress",
            percent: overallPercent,
            message: `Processing with ${this.maxWorkers} workers... Active: ${activeWorkerCount}, Complete: ${this.aggregatedResults.completedWorkers}`,
            processedCount:
                totalProcessed + this.aggregatedResults.totalProcessed,
            uniqueDistribution: combinedDistribution,
            niceNumbers: combinedNiceNumbers.sort(
                (a, b) => a.number - b.number,
            ),
        };
    }

    aggregateWorkerResults(result) {
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

        // Sort nice numbers by value
        this.aggregatedResults.niceNumbers.sort((a, b) => a.number - b.number);

        // Convert distribution map to server format
        const serverDistribution = Array.from(
            this.aggregatedResults.uniqueDistribution.entries(),
        )
            .map(([num_uniques, count]) => ({
                num_uniques: num_uniques,
                count: count,
            }))
            .sort((a, b) => a.num_uniques - b.num_uniques);

        // Get claim_id from any worker result, or from the original claim data
        let claim_id = 0;
        if (
            this.aggregatedResults.niceNumbers.length > 0 &&
            this.aggregatedResults.niceNumbers[0].claim_id
        ) {
            claim_id = this.aggregatedResults.niceNumbers[0].claim_id;
        } else {
            // Get from active jobs or stored claim data
            const firstJob = Array.from(this.activeJobs.values())[0];
            if (firstJob && firstJob.claimData) {
                claim_id = parseInt(firstJob.claimData.claim_id);
            }
        }

        // Get username from any active job
        let username = "anonymous";
        const firstJob = Array.from(this.activeJobs.values())[0];
        if (firstJob && firstJob.username) {
            username = firstJob.username;
        }

        const finalResult = {
            claim_id: claim_id,
            username: username,
            client_version: `3.0.0-wasm-worker-pool-${this.maxWorkers}`,
            unique_distribution: serverDistribution,
            nice_numbers: this.aggregatedResults.niceNumbers,
        };

        if (this.completeCallback) {
            this.completeCallback({
                type: "complete",
                result: finalResult,
                elapsedSeconds: elapsedSeconds,
            });
        }
    }

    resetAggregatedResults() {
        this.aggregatedResults = {
            niceNumbers: [],
            uniqueDistribution: new Map(),
            totalProcessed: 0,
            completedWorkers: 0,
            errors: [],
        };
        this.workerProgress.clear();
        this.lastProgressUpdate = 0;
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

        // Clear active jobs and reset state
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
    }

    async getBenchmarkData() {
        if (!this.isInitialized || this.workers.length === 0) {
            throw new Error("Worker pool not initialized");
        }

        return new Promise((resolve, reject) => {
            const worker = this.workers[0].worker;
            const timeout = setTimeout(() => {
                reject(new Error("Benchmark data timeout"));
            }, 5000);

            const originalHandler = worker.onmessage;
            worker.onmessage = (e) => {
                if (e.data.type === "benchmark_data") {
                    clearTimeout(timeout);
                    worker.onmessage = originalHandler;
                    resolve(e.data.data);
                } else if (originalHandler) {
                    originalHandler(e);
                }
            };

            worker.postMessage({ type: "benchmark" });
        });
    }

    getWorkerCount() {
        return this.maxWorkers;
    }

    setWorkerCount(count) {
        if (count > 0 && count <= (navigator.hardwareConcurrency || 4)) {
            this.maxWorkers = count;
            console.log(`Worker count set to ${count}`);
        }
    }

    isReady() {
        return this.isInitialized && this.workers.every((w) => w.isReady);
    }
}

// Export for use in main thread
window.WorkerPool = WorkerPool;
