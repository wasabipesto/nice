// End-to-end test of the browser client: serves web/search, drives the real
// page in headless Chromium, and runs the offline benchmark on the CPU
// backend and (when WebGPU is available — SwiftShader supplies it in CI
// containers) on the GPU backend, asserting both complete and agree.
//
// Run via ./run.sh (docker + playwright image) or directly with playwright
// installed: node e2e.mjs [--skip-gpu]
import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { extname, join, normalize } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";

const root = normalize(join(fileURLToPath(import.meta.url), "../../../web/search"));
const MIME = {
    ".html": "text/html",
    ".js": "text/javascript",
    ".mjs": "text/javascript",
    ".wasm": "application/wasm",
    ".css": "text/css",
    ".json": "application/json",
};

const server = createServer(async (req, res) => {
    try {
        const path = req.url === "/" ? "/index.html" : req.url.split("?")[0];
        const file = normalize(join(root, path));
        if (!file.startsWith(root)) throw new Error("traversal");
        const body = await readFile(file);
        res.writeHead(200, {
            "Content-Type": MIME[extname(file)] ?? "application/octet-stream",
        });
        res.end(body);
    } catch {
        res.writeHead(404);
        res.end("not found");
    }
});
await new Promise((r) => server.listen(0, "127.0.0.1", r));
const url = `http://127.0.0.1:${server.address().port}/`;
console.log(`serving ${root} at ${url}`);

// These flags are for *this harness's* environment and say nothing about
// what real users need. Headless Chromium in a container brings up no GPU
// by default: measured here, plain defaults and --enable-features=Vulkan
// both expose navigator.gpu while every requestAdapter returns null, and
// --enable-unsafe-webgpu is what makes an adapter appear. The swiftshader
// override then pins the software adapter so the run never depends on host
// GPU drivers.
const browser = await chromium.launch({
    headless: true,
    args: [
        "--enable-unsafe-webgpu",
        "--use-webgpu-adapter=swiftshader",
        "--enable-features=Vulkan",
    ],
});

// Drive one full offline-benchmark run on the given backend and return the
// page's final state (status text, processed count, histogram rows).
async function runBackend(backend) {
    const page = await browser.newPage();
    page.on("console", (m) => {
        if (m.type() === "error") console.log(`[${backend}] page error:`, m.text());
    });
    page.on("pageerror", (e) => console.log(`[${backend}] pageerror:`, e.message));
    await page.goto(url);

    // Wait for the worker pool and the GPU probe to settle.
    await page.waitForSelector("#startBtn:not([disabled])", { timeout: 120000 });
    await page.waitForFunction(
        () =>
            !document
                .getElementById("backendHelp")
                .textContent.includes("Checking"),
        undefined,
        { timeout: 120000 },
    );

    const gpuOffered = await page.evaluate(
        () => document.querySelectorAll("#backendSelect option").length > 1,
    );
    if (backend === "gpu" && !gpuOffered) {
        await page.close();
        return { skipped: "no WebGPU adapter offered" };
    }

    await page.selectOption("#testMode", "true"); // offline benchmark
    await page.selectOption("#backendSelect", backend);
    await page.click("#startBtn");

    // The offline benchmark is one 10M-number base-40 field.
    await page.waitForFunction(
        () => {
            const el = document.getElementById("status");
            return el && /complete|finished|processed/i.test(el.textContent);
        },
        undefined,
        { timeout: 20 * 60 * 1000 },
    );

    const state = await page.evaluate(() => ({
        status: document.getElementById("status").textContent.trim(),
        results: document.getElementById("results").textContent.trim(),
        histogram: window.lastDistribution ?? null,
    }));
    await page.close();
    return state;
}

const skipGpu = process.argv.includes("--skip-gpu");
let failed = false;

console.log("=== CPU backend ===");
const cpu = await runBackend("cpu");
console.log(cpu.status);
if (!/complete/i.test(cpu.status ?? "")) {
    console.log("CPU run did not complete:", cpu);
    failed = true;
}

let gpu = { skipped: "--skip-gpu" };
if (!skipGpu) {
    console.log("=== GPU backend ===");
    gpu = await runBackend("gpu");
    if (gpu.skipped) {
        console.log("GPU skipped:", gpu.skipped);
    } else {
        console.log(gpu.status);
        if (!/complete/i.test(gpu.status ?? "")) {
            console.log("GPU run did not complete:", gpu);
            failed = true;
        }
    }
}

// Cross-backend agreement: the offline benchmark is a fixed field, so the
// two backends' histograms must match exactly.
if (!failed && !gpu.skipped && cpu.histogram && gpu.histogram) {
    const a = JSON.stringify(cpu.histogram);
    const b = JSON.stringify(gpu.histogram);
    if (a === b) {
        console.log("CPU and GPU histograms agree exactly");
    } else {
        console.log("HISTOGRAM MISMATCH\ncpu:", a, "\ngpu:", b);
        failed = true;
    }
}

await browser.close();
server.close();
process.exit(failed ? 1 : 0);
