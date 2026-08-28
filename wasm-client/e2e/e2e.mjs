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
// `--skip-cpu` is for hardware benchmarking: the two backends want very
// different field sizes to measure cleanly (the GPU needs a big one to
// amortise kernel-compile warmup, which the CPU pool would then grind
// through for minutes), so each is timed on its own run.
const skipCpu = process.argv.includes("--skip-cpu");
let failed = false;

let cpu = { skipped: "--skip-cpu" };
if (!skipCpu) {
    console.log("=== CPU backend ===");
    cpu = await runBackend("cpu");
    console.log(cpu.status);
    if (!/complete/i.test(cpu.status ?? "")) {
        console.log("CPU run did not complete:", cpu);
        failed = true;
    }
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
if (!failed && !gpu.skipped && !cpu.skipped && cpu.histogram && gpu.histogram) {
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

// How many submissions to collect before judging the run. More than one
// matters: the pipelined loop must keep claiming and submitting on its own,
// and the claim/submit bookkeeping (distinct ids, prefetch depth) only
// shows up across several fields.
const WANT_SUBMITS = 4;

async function submitShape(backend) {
    const page = await browser.newPage();
    const submissions = []; // { body, raw } in arrival order
    let claimsIssued = 0;
    // The page compiles the wasm once and structured-clones the Module to
    // every worker; if that regresses, each of the ~9 workers fetches its own
    // copy of the 1.9 MB binary and this count jumps with the thread count.
    let wasmFetches = 0;
    page.on("request", (r) => {
        // GET only: logBuild() issues a deliberate HEAD probe for the build
        // banner, which transfers no body.
        if (r.url().endsWith(".wasm") && r.method() === "GET") wasmFetches += 1;
    });
    page.on("pageerror", (e) => console.log(`[submit:${backend}] pageerror:`, e.message));
    await page.route("**/api.nicenumbers.net/**", async (route) => {
        const req = route.request();
        if (req.method() === "OPTIONS") {
            return route.fulfill({ status: 204, headers: CORS });
        }
        const url = req.url();
        if (url.includes("/claim/")) {
            // Same field every time, but a fresh claim_id per claim, so the
            // submissions can be matched 1:1 against what was claimed.
            claimsIssued += 1;
            return route.fulfill({
                status: 200,
                headers: { ...CORS, "Content-Type": "application/json" },
                body: JSON.stringify({
                    ...CLAIM_STUB,
                    claim_id: CLAIM_STUB.claim_id + claimsIssued,
                }),
            });
        }
        if (url.includes("/submit")) {
            // The raw text matters as much as the parsed object: JSON.parse
            // here would round a near-miss number above 2^53 exactly the way
            // the browser did, hiding the bug this checks for.
            const raw = req.postData() ?? "";
            let body;
            try {
                body = req.postDataJSON();
            } catch {
                body = { unparseable: raw };
            }
            submissions.push({ body, raw });
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
    while (submissions.length < WANT_SUBMITS && Date.now() < deadline) {
        await new Promise((r) => setTimeout(r, 200));
    }
    // Snapshot before closing: the loop keeps claiming as the page dies.
    const claimsAtCapture = claimsIssued;
    // The page loops into the next field after a submission; closing here
    // stops it, and an in-flight route can reject as it goes.
    await page.close().catch(() => {});
    return { submissions, claimsAtCapture, wasmFetches };
}

if (!failed) {
    console.log("=== submit payload shape ===");
    const backends = ["cpu", "gpu"].filter(
        (b) => !(b === "cpu" && skipCpu) && !(b === "gpu" && skipGpu),
    );
    for (const backend of backends) {
        const { submissions, claimsAtCapture, wasmFetches, skipped } =
            await submitShape(backend);
        if (skipped) {
            console.log(`[submit:${backend}] skipped: ${skipped}`);
            continue;
        }
        if (!submissions?.length) {
            console.log(`[submit:${backend}] no submission was made`);
            failed = true;
            continue;
        }
        const problems = [];
        const captured = submissions[0].body;
        const raw = submissions[0].raw;
        if (typeof captured.claim_id !== "number") {
            problems.push(
                `claim_id is ${typeof captured.claim_id} (${JSON.stringify(
                    captured.claim_id,
                )}), server expects a number`,
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
        if (wasmFetches !== 1) {
            problems.push(
                `wasm binary fetched ${wasmFetches} times, expected 1 ` +
                    `(the compiled module should be shared across workers)`,
            );
        }
        if (typeof captured.username !== "string") problems.push("username missing");
        if (typeof captured.client_version !== "string") {
            problems.push("client_version missing");
        } else if (/^unknown|^3\.0\.0/.test(captured.client_version)) {
            // 3.0.0 was the hardcoded string the payload carried for three
            // minor versions; the version must now come from the wasm build.
            problems.push(
                `client_version ${captured.client_version} was not read from the wasm build`,
            );
        }

        // The pipelined loop must keep going unattended...
        if (submissions.length < WANT_SUBMITS) {
            problems.push(
                `only ${submissions.length}/${WANT_SUBMITS} submissions arrived`,
            );
        }
        // ...every submission must answer a distinct claim the stub issued...
        const issued = new Set(
            Array.from(
                { length: claimsAtCapture },
                (_, i) => CLAIM_STUB.claim_id + i + 1,
            ),
        );
        const answered = submissions.map((s) => s.body.claim_id);
        if (new Set(answered).size !== answered.length) {
            problems.push(`duplicate claim_ids submitted: ${answered}`);
        }
        for (const id of answered) {
            if (!issued.has(id)) {
                problems.push(`claim_id ${id} was never issued by the stub`);
            }
        }
        // ...and claims must run ahead of submissions: the buffer holding at
        // least one unprocessed claim at capture time is what distinguishes
        // the pipeline from the old claim-submit-claim serial loop.
        if (claimsAtCapture <= submissions.length) {
            problems.push(
                `${claimsAtCapture} claims for ${submissions.length} submissions — ` +
                    `no prefetch happened`,
            );
        }

        if (problems.length) {
            console.log(`[submit:${backend}] BAD PAYLOAD: ${problems.join("; ")}`);
            failed = true;
        } else {
            console.log(
                `[submit:${backend}] ok — ${submissions.length} submissions ` +
                    `answering ${claimsAtCapture} claims (prefetch live), ` +
                    `claim_id numeric, ${dist.length} bins, ` +
                    `near miss ${NEAR_MISS} exact, 1 wasm fetch, ` +
                    `client_version ${captured.client_version}`,
            );
        }
    }
}

// The benchmark suite: same fixed windows as the native sweep (the plan
// comes out of the wasm build), scored in Rust, uploadable. CPU only here —
// the GPU scenarios use 2e8-number windows sized for real hardware, which
// SwiftShader would grind on for many minutes.
async function benchmarkSuite() {
    const page = await browser.newPage();
    let uploaded = null;
    page.on("pageerror", (e) => console.log(`[suite] pageerror:`, e.message));
    await page.route("**/api.nicenumbers.net/**", async (route) => {
        const req = route.request();
        if (req.method() === "OPTIONS") {
            return route.fulfill({ status: 204, headers: CORS });
        }
        const url = req.url();
        if (url.includes("/ping")) {
            return route.fulfill({ status: 200, headers: CORS, body: "pong" });
        }
        if (url.includes("/benchmark")) {
            try {
                uploaded = req.postDataJSON();
            } catch {
                uploaded = { unparseable: req.postData() };
            }
            return route.fulfill({
                status: 200,
                headers: { ...CORS, "Content-Type": "application/json" },
                body: JSON.stringify({ message: "Benchmark stored, thanks!", benchmark_id: 42 }),
            });
        }
        return route.fulfill({ status: 404, headers: CORS, body: "" });
    });
    await page.goto(url);
    await page.waitForSelector("#startBtn:not([disabled])", { timeout: 120000 });

    await page.selectOption("#testMode", "suite");
    await page.selectOption("#backendSelect", "cpu");
    await page.click("#startBtn");

    await page.waitForFunction(
        () => /suite complete/i.test(document.getElementById("status").textContent),
        undefined,
        { timeout: 10 * 60 * 1000 },
    );
    const scoreText = await page.textContent("#nicemarkScore");

    await page.click("#uploadBtn");
    // Wait for the page to process the stub's response, not merely for the
    // request to be captured — the status label lags the POST.
    await page
        .waitForFunction(
            () =>
                /accepted|failed/i.test(
                    document.getElementById("uploadStatus").textContent,
                ),
            undefined,
            { timeout: 60000 },
        )
        .catch(() => {});
    const uploadStatus = await page.textContent("#uploadStatus");
    await page.close().catch(() => {});
    return { scoreText, uploaded, uploadStatus };
}

if (!failed && !skipCpu) {
    console.log("=== benchmark suite (cpu) ===");
    const { scoreText, uploaded, uploadStatus } = await benchmarkSuite();
    const problems = [];
    if (!/^\d+$/.test(scoreText?.trim() ?? "")) {
        problems.push(`NiceMark rendered as ${JSON.stringify(scoreText)}, expected a number`);
    }
    if (!uploaded) {
        problems.push("no upload was posted");
    } else {
        const data = uploaded.data ?? {};
        if (typeof uploaded.username !== "string") problems.push("upload missing username");
        if (data.schema_version !== 1) {
            problems.push(`schema_version ${data.schema_version}, server accepts 1`);
        }
        if (data.config?.platform !== "browser") {
            problems.push("config.platform is not \"browser\"");
        }
        if (!/-wasm-worker$/.test(data.client_version ?? "")) {
            problems.push(`client_version ${data.client_version} lacks the -wasm-worker suffix`);
        }
        const scenarios = data.scenarios ?? [];
        if (scenarios.length !== 3) {
            problems.push(`${scenarios.length} scenarios, expected the 3 detailed ones`);
        }
        if (!scenarios.every((s) => s.rate > 0 && s.repetitions >= 1)) {
            problems.push("a scenario measured no work");
        }
        const solo = scenarios.find((s) => s.key.endsWith("_1t"));
        if (!solo || solo.threads !== 1) {
            problems.push("the single-thread scenario did not run on one worker");
        }
        if (typeof data.score !== "number") problems.push("score missing from report");
    }
    if (!/accepted/i.test(uploadStatus ?? "")) {
        problems.push(`upload status ${JSON.stringify(uploadStatus)} never showed acceptance`);
    }
    if (problems.length) {
        console.log(`[suite] BAD: ${problems.join("; ")}`);
        failed = true;
    } else {
        console.log(
            `[suite] ok — NiceMark ${scoreText.trim()}, ` +
                `${uploaded.data.scenarios.length} scenarios uploaded ` +
                `(platform browser, ${uploaded.data.client_version})`,
        );
    }
}

await browser.close();
server.close();
process.exit(failed ? 1 : 0);
