// GPU-versus-CPU parity for the browser client.
//
// Runs `process_chunk_gpu` (the WebGPU kernel, compiled by this browser for
// this device) and `process_chunk_wasm` (the CPU path) over the same windows
// and compares the two results as raw JSON strings, byte for byte. Nothing is
// parsed before the comparison, so near misses above 2^53 are compared as
// exact digits.
//
// Used by parity.html (open it in any browser with WebGPU) and by the
// end-to-end harness (wasm-client/e2e/e2e.mjs), which drives that page.
// Nothing here is part of the client itself.
import init, * as nice from "./pkg/nice_wasm_client.js";

/** Load the wasm module once and return its exports. */
let loaded = null;
export function loadWasm() {
    loaded ??= init().then(() => nice);
    return loaded;
}

// ---------------------------------------------------------------------------
// Base ranges (common/src/base_range.rs), in BigInt
// ---------------------------------------------------------------------------

function iroot(x, r) {
    if (x < 2n) return x;
    let lo = 0n;
    let hi = 1n;
    while (hi ** r <= x) hi <<= 1n;
    while (hi - lo > 1n) {
        const mid = (lo + hi) >> 1n;
        if (mid ** r <= x) lo = mid;
        else hi = mid;
    }
    return lo;
}

function ceilRoot(x, r) {
    const f = iroot(x, BigInt(r));
    return f ** BigInt(r) === x ? f : f + 1n;
}

/** `[start, end)` of the base's search range, or null (b ≡ 1 mod 5). */
export function baseRange(b) {
    const B = BigInt(b);
    const k = BigInt(Math.floor(b / 5));
    switch (b % 5) {
        case 0:
            return [ceilRoot(B ** (3n * k - 1n), 3), B ** k];
        case 2:
            return [B ** k, ceilRoot(B ** (3n * k + 1n), 3)];
        case 3:
            return [ceilRoot(B ** (3n * k + 1n), 3), ceilRoot(B ** (2n * k + 1n), 2)];
        case 4:
            return [ceilRoot(B ** (2n * k + 1n), 2), ceilRoot(B ** (3n * k + 2n), 3)];
        default:
            return null;
    }
}

const U128_MAX = (1n << 128n) - 1n;

/**
 * The points inside a base's range where a finite-difference addend gains a
 * digit, as the native detailed-mode tests compute them: 2n+1 ≥ b^k (dS),
 * 6n+6 ≥ b^k (ddC) and 3n²+3n+1 ≥ b^k (dC), kept where a window of
 * `width` on either side stays inside the range.
 */
export function gainPoints(b, width) {
    const [rs, re] = baseRange(b);
    const B = BigInt(b);
    const sat = (x, y) => (x > y ? x - y : 0n);
    const ceilDiv = (x, y) => (x + y - 1n) / y;
    const out = [];
    for (let k = 1n; k <= 30n; k++) {
        const bk = B ** k;
        if (bk > U128_MAX) break;
        let lo = 0n;
        let hi = 1n << 64n;
        while (lo < hi) {
            const mid = (lo + hi) >> 1n;
            const v = 3n * mid * mid + 3n * mid + 1n;
            if (v > U128_MAX || v >= bk) hi = mid;
            else lo = mid + 1n;
        }
        const points = [
            ["dS", ceilDiv(sat(bk, 1n), 2n)],
            ["ddC", ceilDiv(sat(bk, 6n), 6n)],
            ["dC", lo],
        ];
        for (const [name, g] of points) {
            if (g > rs + width && g + width < re) out.push({ name, k: Number(k), g });
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// Windows
// ---------------------------------------------------------------------------

// splitmix64 with a fixed seed, so every run picks the same "seeded" windows.
function seededBelow(seed) {
    let s = seed;
    const mask = 0xffffffffffffffffn;
    const next64 = () => {
        s = (s + 0x9e3779b97f4a7c15n) & mask;
        let z = s;
        z = ((z ^ (z >> 30n)) * 0xbf58476d1ce4e5b9n) & mask;
        z = ((z ^ (z >> 27n)) * 0x94d049bb133111ebn) & mask;
        return z ^ (z >> 31n);
    };
    return (n) => ((next64() << 64n) | next64()) % n;
}

// Bases above 25 where Apple's Metal compiler once broke the GPU detailed
// kernel's digit seed (#225); 23 and 25 are covered whole.
const METAL_FAULT_BASES = [27, 28, 29, 30, 49, 50, 52, 53, 54, 55];
// Window lengths that start, end or straddle a GPU thread's run.
const PARTIAL_LENGTHS = [1n, 2n, 61n, 127n, 128n, 129n, 255n, 257n, 100001n];
// Near misses with values above 2^53.
const NEAR_MISS_B49 = 20363742218601559n; // 45 uniques
const NEAR_MISS_B40 = 1917398732423n; // 37 uniques

function win(set, key, base, start, end) {
    return { set, key, base, start: BigInt(start), end: BigInt(end) };
}

/**
 * The full window list, for hardware runs: about 200 windows and 2.5e7
 * numbers over every base from 10 to 64 and two above. Each window is
 * `{ set, key, base, start, end }` with BigInt bounds.
 *
 *   full      bases 10-25 whole (every near miss up to base 25, and 69)
 *   sme       start, middle and end of bases 27-64, 1e5 each
 *   seeded    four fixed pseudo-random 1e5 windows at each METAL_FAULT_BASES
 *   gain      every digit-gain point at bases 27-64, 1e5 centred
 *   boundary  straddling the range start (b40, b50) and end (b40); b10 past its end
 *   partial   PARTIAL_LENGTHS at the b40 and b57 starts
 *   nearmiss  1e5 around a b49 and a b40 near miss above 2^53
 *   above64   b80 and b97 starts, 5e4 each
 *   random    (only with randomPerBase > 0) that many pseudo-random windows
 *             of randomWidth at every base 10-64
 */
export function hardwareWindows({ randomPerBase = 0, randomWidth = 1_000_000n } = {}) {
    const w = [];
    const W = 100_000n;
    for (let b = 10; b <= 25; b++) {
        const r = baseRange(b);
        if (r) w.push(win("full", `b${b}/full`, b, r[0], r[1]));
    }
    for (let b = 27; b <= 64; b++) {
        const r = baseRange(b);
        if (!r) continue;
        const [rs, re] = r;
        const mid = rs + (re - rs) / 2n;
        w.push(win("sme", `b${b}/start`, b, rs, rs + W));
        w.push(win("sme", `b${b}/middle`, b, mid - W / 2n, mid + W / 2n));
        w.push(win("sme", `b${b}/end`, b, re - W, re));
    }
    const below = seededBelow(0x4c32345f6d346677n);
    for (const b of METAL_FAULT_BASES) {
        const [rs, re] = baseRange(b);
        for (let i = 0; i < 4; i++) {
            const s = rs + below(re - rs - W);
            w.push(win("seeded", `b${b}/seeded${i}`, b, s, s + W));
        }
    }
    for (let b = 27; b <= 64; b++) {
        if (!baseRange(b)) continue;
        for (const { name, k, g } of gainPoints(b, W)) {
            w.push(win("gain", `b${b}/${name}-k${k}@${g}`, b, g - W / 2n, g + W / 2n));
        }
    }
    for (const b of [40, 50]) {
        const [rs] = baseRange(b);
        w.push(win("boundary", `b${b}/straddle-start`, b, rs - 50_000n, rs + 50_000n));
    }
    const [, re40] = baseRange(40);
    w.push(win("boundary", "b40/straddle-end", 40, re40 - 50_000n, re40 + 50_000n));
    w.push(win("boundary", "b10/[47,200)", 10, 47n, 200n));
    for (const b of [40, 57]) {
        const [rs] = baseRange(b);
        for (const len of PARTIAL_LENGTHS) {
            w.push(win("partial", `b${b}/len${len}`, b, rs, rs + len));
        }
    }
    w.push(win("nearmiss", "b49/nearmiss", 49, NEAR_MISS_B49 - 50_000n, NEAR_MISS_B49 + 50_000n));
    w.push(win("nearmiss", "b40/nearmiss", 40, NEAR_MISS_B40 - 50_000n, NEAR_MISS_B40 + 50_000n));
    for (const b of [80, 97]) {
        const [rs] = baseRange(b);
        w.push(win("above64", `b${b}/start`, b, rs, rs + 50_000n));
    }
    if (randomPerBase > 0) {
        const pick = seededBelow(0x4c32345f72616e64n);
        const width = BigInt(randomWidth);
        for (let b = 10; b <= 64; b++) {
            const r = baseRange(b);
            if (!r) continue;
            const [rs, re] = r;
            const span = re - rs;
            const ww = span > width ? width : span;
            for (let i = 0; i < randomPerBase; i++) {
                const s = rs + (span > ww ? pick(span - ww) : 0n);
                w.push(win("random", `b${b}/random${i}`, b, s, s + ww));
            }
        }
    }
    return w;
}

/**
 * The CI subset: 13 windows, 1.1e6 numbers, about 11 s on a software
 * adapter (SwiftShader), most of it compiling one kernel per base.
 */
export function ciWindows() {
    const all = hardwareWindows();
    const pick = (pred, what) => {
        const found = all.find(pred);
        if (!found) throw new Error(`CI window ${what} is missing from the hardware list`);
        return { ...found, set: "ci" };
    };
    const byKey = (key) => pick((w) => w.key === key, key);
    const gain = (base, name) =>
        pick((w) => w.set === "gain" && w.base === base && w.key.includes(`/${name}-`), `b${base} ${name}`);
    const out = [
        byKey("b10/[47,200)"),
        byKey("b23/full"),
        byKey("b40/start"),
        byKey("b40/straddle-start"),
        gain(40, "dS"),
        byKey("b40/len1"),
        byKey("b40/len61"),
        byKey("b40/len129"),
        byKey("b49/nearmiss"),
        gain(50, "dC"),
    ];
    for (const b of [57, 64, 80]) {
        const [rs] = baseRange(b);
        out.push(win("ci", `b${b}/start-2e4`, b, rs, rs + 20_000n));
    }
    return out;
}

/** The window list by name: "ci" or "hw". */
export function windowsFor(subset, opts = {}) {
    if (subset === "ci") return ciWindows();
    if (subset === "hw") return hardwareWindows(opts);
    throw new Error(`unknown parity subset ${JSON.stringify(subset)} (ci or hw)`);
}

/**
 * The wasm build's own range starts (via the benchmark plan) must agree with
 * the BigInt ranges above, or the windows would not be where they claim.
 */
export function checkRangesAgainstPlan(wasm) {
    const plan = JSON.parse(wasm.benchmark_plan());
    for (const s of plan.scenarios) {
        const ours = baseRange(s.base)?.[0];
        if (BigInt(s.start) !== ours) {
            throw new Error(`the wasm plan starts ${s.key} at ${s.start}; baseRange(${s.base}) says ${ours}`);
        }
    }
    return plan;
}

// ---------------------------------------------------------------------------
// Adapter
// ---------------------------------------------------------------------------

/** Adapter names that mean software rendering rather than a GPU. */
export const SOFTWARE_ADAPTER = /swiftshader|llvmpipe|lavapipe|software|basic render/i;

/**
 * The adapter as this page sees it, the name the wasm reports from
 * `gpu_init()`, and the build. Requested as CubeCL requests it (high
 * performance). Never throws: no adapter is `{ adapter: false }`.
 */
export async function adapterRecord(wasm) {
    const rec = { navigator_gpu: !!globalThis.navigator?.gpu, user_agent: globalThis.navigator?.userAgent };
    try {
        const a = await navigator.gpu?.requestAdapter({ powerPreference: "high-performance" });
        rec.adapter = !!a;
        if (a) {
            const i = a.info ?? (await a.requestAdapterInfo?.());
            rec.vendor = i?.vendor;
            rec.architecture = i?.architecture;
            rec.device = i?.device;
            rec.description = i?.description;
            // GPUAdapterInfo.isFallbackAdapter since Chrome 136; the adapter
            // attribute before that.
            rec.fallback = i?.isFallbackAdapter ?? a.isFallbackAdapter;
            rec.subgroup_sizes = i?.subgroupMinSize ? [i.subgroupMinSize, i.subgroupMaxSize] : undefined;
        }
    } catch (e) {
        rec.adapter = false;
        rec.error = String(e);
    }
    try {
        rec.build_sha = wasm.build_sha();
        rec.client_version = wasm.client_version();
        rec.gpu_build_info = wasm.gpu_build_info();
    } catch (e) {
        rec.build_error = String(e);
    }
    if (rec.adapter) {
        try {
            rec.gpu_init_name = await wasm.gpu_init();
        } catch (e) {
            rec.gpu_init_error = String(e);
        }
    }
    return rec;
}

/** One line for logs and the page. */
export function describeAdapter(rec) {
    if (!rec.adapter) return `no WebGPU adapter${rec.error ? ` (${rec.error})` : ""}`;
    const q = (v) => JSON.stringify(v ?? null);
    return (
        `vendor=${q(rec.vendor)} architecture=${q(rec.architecture)} device=${q(rec.device)} ` +
        `description=${q(rec.description)} fallback=${rec.fallback} ` +
        `gpu_init=${q(rec.gpu_init_name ?? rec.gpu_init_error)}`
    );
}

/** Whether the adapter is a software rasterizer or the browser's fallback. */
export function isSoftwareAdapter(rec) {
    return (
        rec.fallback === true ||
        ["vendor", "architecture", "device", "description", "gpu_init_name"].some(
            (k) => typeof rec[k] === "string" && SOFTWARE_ADAPTER.test(rec[k]),
        )
    );
}

/**
 * Time `process_chunk_gpu` over `[start, start + size)`: `warm` untimed
 * calls (the first compiles the kernel), then one timed call. Returns
 * `{ first_secs, warm_secs, rate }`.
 */
export async function timeGpu(wasm, { base, start, size, warm = 1 }) {
    const s = BigInt(start).toString();
    const e = (BigInt(start) + BigInt(size)).toString();
    let first_secs = null;
    for (let i = 0; i < warm; i++) {
        const t0 = performance.now();
        await wasm.process_chunk_gpu(s, e, base);
        if (i === 0) first_secs = (performance.now() - t0) / 1000;
    }
    const t0 = performance.now();
    await wasm.process_chunk_gpu(s, e, base);
    const warm_secs = (performance.now() - t0) / 1000;
    return { base, start: s, size: String(size), first_secs, warm_secs, rate: Number(size) / warm_secs };
}

// ---------------------------------------------------------------------------
// Comparison
// ---------------------------------------------------------------------------

/** JSON with every bare integer quoted, so values above 2^53 parse exactly. */
export function parseExact(json) {
    return JSON.parse(json.replace(/([:[,]\s*)(-?\d+)(?=\s*[,}\]])/g, '$1"$2"'));
}

/**
 * Where two results first differ, as one readable line (the field, then both
 * values), plus every differing bin and near miss. Fields are compared in
 * the order the client serialises them: the histogram, then the near misses.
 */
export function describeDifference(gpuJson, cpuJson) {
    let g;
    let c;
    try {
        g = parseExact(gpuJson);
        c = parseExact(cpuJson);
    } catch (e) {
        return { first: `unparseable result: ${e}`, all: [] };
    }
    const all = [];
    const gd = new Map((g.distribution ?? []).map((d) => [Number(d.num_uniques), d.count]));
    const cd = new Map((c.distribution ?? []).map((d) => [Number(d.num_uniques), d.count]));
    const bins = [...new Set([...gd.keys(), ...cd.keys()])].sort((a, b) => a - b);
    for (const k of bins) {
        if (gd.get(k) !== cd.get(k)) {
            all.push(`distribution[num_uniques=${k}].count: gpu ${gd.get(k) ?? "absent"}, cpu ${cd.get(k) ?? "absent"}`);
        }
    }
    const gn = g.nice_numbers ?? [];
    const cn = c.nice_numbers ?? [];
    for (let i = 0; i < Math.max(gn.length, cn.length); i++) {
        const a = gn[i] ? `${gn[i].number}:${gn[i].num_uniques}` : "absent";
        const b = cn[i] ? `${cn[i].number}:${cn[i].num_uniques}` : "absent";
        if (a !== b) all.push(`nice_numbers[${i}] (number:num_uniques): gpu ${a}, cpu ${b}`);
    }
    if (!all.length) {
        // Same content, so the serialisation differs: show the first byte.
        let i = 0;
        while (i < gpuJson.length && gpuJson[i] === cpuJson[i]) i++;
        all.push(
            `byte ${i}: gpu …${gpuJson.slice(Math.max(0, i - 30), i + 30)}…, cpu …${cpuJson.slice(Math.max(0, i - 30), i + 30)}…`,
        );
    }
    return { first: all[0], all };
}

async function sha256(text) {
    if (!globalThis.crypto?.subtle) return null;
    const d = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text));
    return [...new Uint8Array(d)].map((x) => x.toString(16).padStart(2, "0")).join("");
}

/**
 * One window through both paths. The record says whether the raw strings
 * are identical, and on any difference which field differs first. The CPU
 * histogram must also count every number of the window, so two empty
 * results cannot pass as agreement.
 */
export async function compareWindow(wasm, w) {
    const s = w.start.toString();
    const e = w.end.toString();
    const n = w.end - w.start;
    const out = { set: w.set, key: w.key, base: w.base, start: s, end: e, numbers: n.toString() };
    let gpu = null;
    let cpu = null;
    let t = performance.now();
    try {
        gpu = await wasm.process_chunk_gpu(s, e, w.base);
    } catch (err) {
        out.gpu_error = String(err);
    }
    out.gpu_secs = (performance.now() - t) / 1000;
    t = performance.now();
    try {
        cpu = wasm.process_chunk_wasm(s, e, w.base);
    } catch (err) {
        out.cpu_error = String(err);
    }
    out.cpu_secs = (performance.now() - t) / 1000;

    out.identical = gpu !== null && cpu !== null && gpu === cpu;
    out.problems = [];
    if (out.gpu_error) out.problems.push(`GPU path failed: ${out.gpu_error}`);
    if (out.cpu_error) out.problems.push(`CPU path failed: ${out.cpu_error}`);
    out.near_misses = [];
    if (cpu !== null) {
        const p = parseExact(cpu);
        out.near_misses = (p.nice_numbers ?? []).map((x) => `${x.number}:${x.num_uniques}`);
        const counted = (p.distribution ?? []).reduce((a, d) => a + BigInt(d.count), 0n);
        if (counted !== n) out.problems.push(`CPU histogram counts ${counted} numbers; the window has ${n}`);
        out.bytes = cpu.length;
        out.sha256 = await sha256(cpu);
    }
    if (gpu !== null && cpu !== null && !out.identical) {
        const d = describeDifference(gpu, cpu);
        out.problems.push(`results differ, first at ${d.first}`);
        out.differences = d.all.slice(0, 20);
        out.gpu_json = gpu;
        out.cpu_json = cpu;
    }
    out.ok = out.identical && out.problems.length === 0;
    return out;
}

/**
 * Every window from `from` on, in order. `onWindow(record, index)` sees each
 * result as it lands; `shouldStop()` ends the run early. Yields to the event
 * loop between windows so a page stays responsive.
 */
export async function runParity(wasm, windows, { onWindow = () => {}, shouldStop = () => false, from = 0 } = {}) {
    const t0 = performance.now();
    let numbers = 0n;
    let nearMisses = 0;
    let failed = 0;
    let ran = 0;
    for (let i = from; i < windows.length && !shouldStop(); i++) {
        const rec = await compareWindow(wasm, windows[i]);
        ran++;
        numbers += BigInt(rec.numbers);
        nearMisses += rec.near_misses.length;
        if (!rec.ok) failed++;
        await onWindow(rec, i);
        await new Promise((r) => setTimeout(r, 0));
    }
    return {
        windows: ran,
        numbers: numbers.toString(),
        near_misses: nearMisses,
        failed,
        wall_secs: (performance.now() - t0) / 1000,
    };
}
