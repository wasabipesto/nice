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
import { chromium, firefox } from "playwright";

// Defaults to the repo's page; NICE_E2E_ROOT points it at a copy, which is
// how the hardware runs work — the harness ships to a machine with a GPU
// without dragging the whole repo along.
const root = process.env.NICE_E2E_ROOT
    ? normalize(process.env.NICE_E2E_ROOT)
    : normalize(join(fileURLToPath(import.meta.url), "../../../web/search"));
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

// Default (CI) mode pins a software adapter so the run never depends on
// host GPU drivers. These flags are for *this harness's* environment and
// say nothing about what real users need: headless Chromium in a container
// brings up no GPU by default — measured here, plain defaults and
// --enable-features=Vulkan both expose navigator.gpu while every
// requestAdapter returns null, and --enable-unsafe-webgpu is what makes an
// adapter appear at all.
//
// `--hw` instead launches the browser exactly as a user's would be, with no
// GPU flags whatsoever, so the run exercises the platform's real adapter.
// That is the only mode that can report a meaningful rate, and the only one
// that proves the stack works on hardware someone actually has.
const hardware = process.argv.includes("--hw");
// `channel: "chromium"` matters in hardware mode: Playwright's default
// headless binary is chrome-headless-shell, which ships without GPU support
// at all, so it reports no adapter on a perfectly good GPU. The full
// Chromium in new-headless mode is what a user's browser actually is.
//
// `--firefox` runs the same page in Firefox instead. Firefox gates WebGPU
// behind three prefs on Linux: the feature, WebGPU inside workers (which is
// where this client runs it), and the driver blocklist — Firefox ships a
// conservative adapter blocklist on Linux, and a blocklisted adapter makes
// requestAdapter return null, which is indistinguishable from "no hardware"
// unless the pref is cleared. None of the three is on by default in a
// release channel yet, so this mode is for checking the stack works there,
// not for CI.
const useFirefox = process.argv.includes("--firefox");
const browser = useFirefox
    ? await firefox.launch({
          headless: true,
          firefoxUserPrefs: {
              "dom.webgpu.enabled": true,
              "dom.webgpu.workers.enabled": true,
              "gfx.webgpu.ignore-blocklist": true,
          },
      })
    : await chromium.launch({
          headless: true,
          ...(hardware ? { channel: "chromium" } : {}),
          args: hardware
              ? []
              : [
                    "--enable-unsafe-webgpu",
                    "--use-webgpu-adapter=swiftshader",
                    "--enable-features=Vulkan",
                ],
      });
console.log(
    `browser: ${useFirefox ? "firefox" : "chromium"}, adapter: ${
        hardware || useFirefox ? "real" : "pinned software"
    }`,
);

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
        const why = await page.textContent("#backendHelp");
        await page.close();
        return { skipped: `no WebGPU adapter offered — page says: ${why?.trim()}` };
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
        adapter:
            [...document.querySelectorAll("#backendSelect option")]
                .find((o) => o.value === "gpu")
                ?.textContent.trim() ?? null,
    }));
    const rate = /Rate: [\d.]+ \(([\d.e+]+)\) numbers\/second/.exec(
        state.results,
    )?.[1];
    if (rate) console.log(`[${backend}] rate: ${rate} numbers/second`);
    if (backend === "gpu" && state.adapter) {
        console.log(`[gpu] adapter: ${state.adapter}`);
    }
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

// The payload the client actually submits. Everything above runs the
// offline benchmark, which never submits, so none of it covers the shape of
// a real submission — and the CPU and GPU paths assemble that payload in
// different files, which is exactly how a defect can land on one backend
// only. (It did: the GPU path sent claim_id as a quoted string and the
// server answered 400, while the CPU path coerced it to a number and
// worked.) This claims a small field from a stubbed server, captures the
// POST, and checks what each backend produced.
// A real base-49 field window, chosen because it contains a known near miss
// whose value is above 2^53: 20363742218601559 has 45 unique digits, and a
// double cannot hold it (the nearest one is ...560). Submitting that field is
// what caught the precision bug, so the stub reproduces the conditions.
const NEAR_MISS = "20363742218601559";
const CLAIM_STUB = {
    claim_id: 245749450, // a number, as the real API sends it
    base: 49,
    range_start: "20363742218551559",
    range_end: "20363742218651559", // 100k numbers around the near miss
    range_size: "100000",
};
const CORS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "*",
    "Access-Control-Allow-Methods": "*",
};

async function submitShape(backend) {
    const page = await browser.newPage();
    let captured = null;
    let raw = "";
    page.on("pageerror", (e) => console.log(`[submit:${backend}] pageerror:`, e.message));
    await page.route("**/api.nicenumbers.net/**", async (route) => {
        const req = route.request();
        if (req.method() === "OPTIONS") {
            return route.fulfill({ status: 204, headers: CORS });
        }
        const url = req.url();
        if (url.includes("/claim/")) {
            return route.fulfill({
                status: 200,
                headers: { ...CORS, "Content-Type": "application/json" },
                body: JSON.stringify(CLAIM_STUB),
            });
        }
        if (url.includes("/submit")) {
            if (captured === null) {
                // The raw text matters as much as the parsed object: JSON.parse
                // here would round a near-miss number above 2^53 exactly the way
                // the browser did, hiding the bug this checks for.
                raw = req.postData() ?? "";
                try {
                    captured = req.postDataJSON();
                } catch {
                    captured = { unparseable: raw };
                }
            }
            return route.fulfill({ status: 200, headers: CORS, body: "accepted" });
        }
        return route.fulfill({ status: 404, headers: CORS, body: "" });
    });
    await page.goto(url);
    await page.waitForSelector("#startBtn:not([disabled])", { timeout: 120000 });
    await page.waitForFunction(
        () => !document.getElementById("backendHelp").textContent.includes("Checking"),
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

    await page.selectOption("#testMode", "false"); // live mode: claims and submits
    await page.selectOption("#backendSelect", backend);
    await page.click("#startBtn");

    const deadline = Date.now() + 10 * 60 * 1000;
    while (captured === null && Date.now() < deadline) {
        await new Promise((r) => setTimeout(r, 200));
    }
    // The page loops into the next field after a submission; closing here
    // stops it, and an in-flight route can reject as it goes.
    await page.close().catch(() => {});
    return { captured, raw };
}

if (!failed) {
    console.log("=== submit payload shape ===");
    for (const backend of skipGpu ? ["cpu"] : ["cpu", "gpu"]) {
        const { captured, raw, skipped } = await submitShape(backend);
        if (skipped) {
            console.log(`[submit:${backend}] skipped: ${skipped}`);
            continue;
        }
        if (!captured) {
            console.log(`[submit:${backend}] no submission was made`);
            failed = true;
            continue;
        }
        const problems = [];
        if (typeof captured.claim_id !== "number") {
            problems.push(
                `claim_id is ${typeof captured.claim_id} (${JSON.stringify(
                    captured.claim_id,
                )}), server expects a number`,
            );
        } else if (captured.claim_id !== CLAIM_STUB.claim_id) {
            // Only worth reporting once the type is right; otherwise the
            // message reads as "245749450 != 245749450".
            problems.push(
                `claim_id ${captured.claim_id} != claimed ${CLAIM_STUB.claim_id}`,
            );
        }
        const dist = captured.unique_distribution;
        if (!Array.isArray(dist) || dist.length === 0) {
            problems.push("unique_distribution missing or empty");
        } else if (!dist.some((d) => d.count > 0)) {
            problems.push("unique_distribution is all zeroes");
        }
        // Exact-value check against the raw body, not the parsed object.
        if (!raw.includes(`"number":${NEAR_MISS}`)) {
            const seen = /"number":\s*"?(\d+)"?/.exec(raw)?.[1] ?? "(none)";
            problems.push(
                `near miss submitted as ${seen}, expected the exact integer ` +
                    `${NEAR_MISS} (a double cannot hold it)`,
            );
        }
        if (typeof captured.username !== "string") problems.push("username missing");
        if (typeof captured.client_version !== "string") {
            problems.push("client_version missing");
        }
        if (problems.length) {
            console.log(`[submit:${backend}] BAD PAYLOAD: ${problems.join("; ")}`);
            failed = true;
        } else {
            console.log(
                `[submit:${backend}] ok — claim_id ${captured.claim_id} (number), ` +
                    `${dist.length} bins, near miss ${NEAR_MISS} exact, ` +
                    `client_version ${captured.client_version}`,
            );
        }
    }
}

await browser.close();
server.close();
process.exit(failed ? 1 : 0);
