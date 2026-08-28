"""Throughput estimation from the benchmark corpus.

A direct port of common/src/estimator.rs. The controller used to reach the
API's POST /estimate once per offer per mode — ~400 round trips a tick,
against a corpus the API truncated to its most recent 2000 reports. Matching
locally removes both limits: the whole corpus is available, and a tick's
estimates cost no network at all.

Kept close to the Rust so the two can be diffed by eye: same stage names,
same confidence numbers, same widening factors, same order of checks. Pure
functions over decoded samples; the corpus sync lives in the controller.

The two must carry the same CPU identity rule: see `cpu_match_key` here and
its counterpart in common/src/estimator.rs. Comparing the raw model strings
instead silently fails whenever Vast's listing string and the client's
/proc/cpuinfo string spell the same chip differently, which is essentially
every Intel part; that function carries the measurement.
"""

from __future__ import annotations

import math

# Stage chain, in the order match_stage tries them. GPU requests walk the GPU
# chain and stop at `floor`; CPU requests walk the CPU chain.
DETAIL_TOKENS = ("sxm", "sxm2", "sxm3", "sxm4", "sxm5", "pcie", "nvl")

BENCHMARK_SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# Normalization


def normalize_model(raw):
    """Lowercase, strip registered/trademark noise, collapse whitespace."""
    s = raw.lower().replace("®", " ").replace("™", " ")
    s = s.replace("(r)", " ").replace("(tm)", " ")
    return " ".join(s.split())


def normalize_gpu_model(raw):
    """`normalize_model` plus the vendor prefixes and VRAM suffixes that Vast
    and CUDA disagree about ("NVIDIA GeForce RTX 3080" vs "RTX 3080")."""
    tokens = normalize_model(raw.replace("-", " ")).split()
    out = []
    for t in tokens:
        if t in ("nvidia", "geforce"):
            continue
        if t.endswith("gb") and t[:-2].isdigit() and t[:-2]:
            continue
        out.append(t)
    return " ".join(out)


def gpu_models_match(a, b):
    """Equal once form-factor detail is discounted: the shorter token list
    must prefix the longer, and every extra token must be a form factor.
    So `a100 sxm4` matches `a100`, but `3060 ti` never matches `3060`."""
    ta, tb = a.split(), b.split()
    short, long = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
    if long[: len(short)] != short:
        return False
    return all(t in DETAIL_TOKENS for t in long[len(short) :])


# Vendor and marketing tokens that appear on one side of a CPU name but not
# the other. Vast lists "Xeon(R) E5-2680 v4" while the client reads
# "Intel(R) Xeon(R) CPU E5-2680 v4 @ 2.40GHz" out of /proc/cpuinfo, so a plain
# string comparison misses the two halves of the same machine.
_CPU_NOISE = frozenset(
    ["intel", "amd", "genuine", "authentic", "cpu", "processor", "core", "with",
     "radeon", "graphics", "ryzen", "th", "gen"]
)


def cpu_match_key(raw):
    """Identity of a CPU across the two places we read its name.

    Drops the clock suffix and the vendor/marketing tokens either source may
    or may not carry, then keeps the first two remaining tokens:
    "xeon e5-2680", "epyc 7763", "i5-11400".

    Measured over 1378 instances where we know both the offer string and the
    benchmark that machine went on to upload, 55% failed to match themselves
    on the raw strings and 0% fail on this key. Held-out prediction over 2538
    cases: median error 3.4% -> 2.4% and the >25% miss rate 11.0% -> 9.6%,
    entirely from Intel parts — AMD names already agreed and score
    bit-identically either way.
    """
    text = normalize_model(raw or "?").split("@")[0]
    tokens = [t for t in text.replace("(", " ").replace(")", " ").split()
              if t not in _CPU_NOISE and not t.endswith("ghz")]
    return " ".join(tokens[:2]) if tokens else "?"


def family_key(normalized_model):
    """Coarse CPU family: first two normalized tokens ("amd epyc")."""
    return " ".join(normalized_model.split()[:2])


def mode_string(raw):
    """Map a user-facing mode name onto the report's SearchMode Display form."""
    key = raw.lower().replace("-", "")
    return {"detailed": "Detailed", "niceonly": "Nice-only"}.get(key)


# ---------------------------------------------------------------------------
# Corpus decoding


class Sample:
    """One decoded benchmark report. `scenarios` is a list of dicts with
    key/base/threads/rate."""

    __slots__ = ("client_version", "gpu", "mode", "threads", "cpu_model",
                 "cpu_key", "gpu_model", "scenarios")

    def __init__(self, client_version, gpu, mode, threads, cpu_model, gpu_model, scenarios):
        self.client_version = client_version
        self.gpu = gpu
        self.mode = mode
        self.threads = threads
        self.cpu_model = cpu_model
        # Derived, not stored upstream: cheap here, and it keeps the per-offer
        # matching loop from re-deriving it for every sample in the pool.
        self.cpu_key = cpu_match_key(cpu_model) if cpu_model else None
        self.gpu_model = gpu_model
        self.scenarios = scenarios


def decode_sample(client_version, data):
    """Decode one stored report, or None if it isn't usable. Mirrors the Rust:
    schema version 1 only, scenarios with a positive rate only, and a report
    with no usable scenario is dropped entirely."""
    if not isinstance(data, dict) or data.get("schema_version") != BENCHMARK_SCHEMA_VERSION:
        return None
    config = data.get("config")
    hardware = data.get("hardware")
    raw_scenarios = data.get("scenarios")
    if not isinstance(config, dict) or not isinstance(hardware, dict):
        return None
    # Browser-suite reports are stored for their own sake but stay out of
    # the estimator corpus: wasm rates are not native rates, and a browser
    # cannot name its CPU, so these samples would land in the coarse
    # fallback buckets and drag native estimates down. Native reports carry
    # no `platform` key at all. Mirrors the Rust.
    if config.get("platform") == "browser":
        return None
    if not isinstance(raw_scenarios, list):
        return None

    scenarios = []
    for s in raw_scenarios:
        if not isinstance(s, dict):
            continue
        rate = s.get("rate")
        key = s.get("key")
        base = s.get("base")
        threads = s.get("threads")
        if not isinstance(rate, (int, float)) or isinstance(rate, bool) or rate <= 0:
            continue
        if not isinstance(key, str):
            continue
        if not isinstance(base, int) or isinstance(base, bool):
            continue
        if not isinstance(threads, int) or isinstance(threads, bool):
            continue
        scenarios.append({"key": key, "base": base, "threads": threads, "rate": float(rate)})
    if not scenarios:
        return None

    gpu = config.get("gpu")
    mode = config.get("mode")
    threads = config.get("threads")
    if not isinstance(gpu, bool) or not isinstance(mode, str):
        return None
    if not isinstance(threads, int) or isinstance(threads, bool):
        return None

    cpu_model = hardware.get("cpu_model")
    gpu_model = hardware.get("gpu_model")
    return Sample(
        client_version=client_version,
        gpu=gpu,
        mode=mode,
        threads=threads,
        cpu_model=normalize_model(cpu_model) if isinstance(cpu_model, str) else None,
        gpu_model=normalize_gpu_model(gpu_model) if isinstance(gpu_model, str) else None,
        scenarios=scenarios,
    )


# ---------------------------------------------------------------------------
# Matching


def quantile(sorted_values, frac):
    """Interpolated quantile of an already-sorted list."""
    if not sorted_values:
        return 0.0
    pos = frac * (len(sorted_values) - 1)
    lo = math.floor(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    t = pos - lo
    return sorted_values[lo] * (1.0 - t) + sorted_values[hi] * t


def thread_scale(sample, target):
    """Rate multiplier taking a sample's multi-thread scenarios to `target`
    threads. Linear in thread count, capped by perfect scaling from the
    sample's own single-thread anchor. None when the anchor pair is absent."""
    anchor_multi = next(
        (s for s in sample.scenarios if not s["key"].endswith("_1t") and s["key"].startswith("b50")),
        None,
    )
    anchor_single = next((s for s in sample.scenarios if s["key"].endswith("_1t")), None)
    if anchor_multi is None or anchor_single is None:
        return None
    source = float(max(sample.threads, 1))
    target_f = float(max(target, 1))
    linear = target_f / source
    ceiling = (anchor_single["rate"] * target_f) / anchor_multi["rate"]
    return max(min(linear, ceiling), 0.01)


class StageMatch:
    __slots__ = ("stage", "confidence", "rows", "spread_widening", "notes")

    def __init__(self, stage, confidence, rows, spread_widening, notes):
        self.stage = stage
        self.confidence = confidence
        self.rows = rows  # list of (Sample, rate multiplier)
        self.spread_widening = spread_widening
        self.notes = notes


def _scaled_stage(candidates, threads, stage, base_confidence, widening):
    """Thread-scaled stage, discounting confidence by how far the requested
    thread count sits from the samples that back it."""
    rows = []
    max_distance = 0.0
    missing_anchor = False
    for s in candidates:
        scale = thread_scale(s, threads)
        if scale is None:
            missing_anchor = True
            continue
        distance = abs(math.log2(float(max(threads, 1)) / float(max(s.threads, 1))))
        max_distance = max(max_distance, distance)
        rows.append((s, scale))
    penalty = int(min(max_distance * 15.0, 30.0))
    notes = [f"rates scaled to {threads} threads from measured configurations"]
    if missing_anchor:
        notes.append("some samples lacked a single-thread anchor and were skipped")
    return StageMatch(stage, max(base_confidence - penalty, 5), rows, widening, notes)


def match_stage(samples, inp):
    """Pick the best-supported stage for this request. `inp` is a dict with
    gpu, mode, threads, cpu_model, gpu_model."""
    pool = [s for s in samples if s.gpu == inp["gpu"] and s.mode == inp["mode"]]
    if not pool:
        return StageMatch("none", 0, [], 1.0, ["no benchmark data for this mode/device class"])

    want_cpu = normalize_model(inp["cpu_model"]) if inp.get("cpu_model") else None
    want_key = cpu_match_key(inp["cpu_model"]) if inp.get("cpu_model") else None
    want_gpu = normalize_gpu_model(inp["gpu_model"]) if inp.get("gpu_model") else None
    threads = inp.get("threads")

    if inp["gpu"]:
        if want_gpu is not None:
            same_gpu = [s for s in pool if s.gpu_model and gpu_models_match(s.gpu_model, want_gpu)]
            if same_gpu:
                if want_cpu is not None:
                    exact = [(s, 1.0) for s in same_gpu if s.cpu_key == want_key]
                    if exact:
                        return StageMatch("exact", 85, exact, 1.0, [])
                if threads is not None:
                    similar = [
                        (s, 1.0)
                        for s in same_gpu
                        if 0.5 <= (float(max(s.threads, 1)) / float(max(threads, 1))) <= 2.0
                    ]
                    if similar:
                        return StageMatch(
                            "same-gpu-similar-cpu", 65, similar, 1.5,
                            ["GPU nice-only throughput is bounded by min(GPU kernel, CPU MSD "
                             "production); CPU details differ from samples"],
                        )
                return StageMatch(
                    "same-gpu", 50, [(s, 1.0) for s in same_gpu], 2.0,
                    ["samples share the GPU but not the CPU; the CPU side (MSD production) "
                     "may bind for nice-only"],
                )
        return StageMatch(
            "floor", 15, [(s, 1.0) for s in pool], 3.0,
            ["no samples for this GPU model; using all GPU data"],
        )

    # CPU chain.
    if want_cpu is not None:
        same_cpu = [s for s in pool if s.cpu_key == want_key]
        if same_cpu:
            if threads is not None:
                exact = [(s, 1.0) for s in same_cpu if s.threads == threads]
                if exact:
                    return StageMatch("exact", 85, exact, 1.0, [])
                return _scaled_stage(same_cpu, threads, "same-cpu-scaled", 60, 1.5)
            return StageMatch(
                "same-cpu-scaled", 45, [(s, 1.0) for s in same_cpu], 1.5,
                ["no target thread count given; rates unscaled"],
            )
        family = family_key(want_cpu)
        same_family = [s for s in pool if s.cpu_model and family_key(s.cpu_model) == family]
        if same_family:
            if threads is not None:
                return _scaled_stage(same_family, threads, "cpu-family-scaled", 40, 2.0)
            return StageMatch(
                "cpu-family-scaled", 30, [(s, 1.0) for s in same_family], 2.0,
                ["no target thread count given; rates unscaled"],
            )
    return StageMatch(
        "floor", 15, [(s, 1.0) for s in pool], 3.0,
        ["no samples for this CPU model or family; using all CPU data"],
    )


# ---------------------------------------------------------------------------
# Estimation


def estimate(samples, inp):
    """Produce an estimate. `samples` is the decoded corpus; order is
    irrelevant. Returns a dict shaped like the API's /estimate response."""
    matched = match_stage(samples, inp)

    versions_used = sorted({s.client_version for s, _ in matched.rows})
    want_version = inp.get("client_version")
    if want_version and matched.rows and want_version not in versions_used:
        matched.confidence = max(matched.confidence - 15, 5)
        matched.notes.append(
            f"no samples from client version {want_version}; estimates come from {versions_used}"
        )

    # Group thread-scaled rates by scenario key, preserving first-seen order
    # so the output matches the Rust before its final sort.
    by_key = {}
    order = []
    want_base = inp.get("base")
    for sample, scale in matched.rows:
        for scenario in sample.scenarios:
            if want_base is not None and scenario["base"] != want_base:
                continue
            key = scenario["key"]
            # Single-thread scenarios are already per-thread; never scaled.
            effective = scenario["rate"] if key.endswith("_1t") else scenario["rate"] * scale
            if key not in by_key:
                by_key[key] = (scenario["base"], [])
                order.append(key)
            by_key[key][1].append(effective)

    scenarios = []
    for key in order:
        base, rates = by_key[key]
        rates.sort()
        p50 = quantile(rates, 0.5)
        iqr_low = p50 - quantile(rates, 0.25)
        iqr_high = quantile(rates, 0.75) - p50
        scenarios.append({
            "key": key,
            "base": base,
            "samples": len(rates),
            "rate_p25": max(p50 - iqr_low * matched.spread_widening, p50 * 0.05),
            "rate_p50": p50,
            "rate_p75": p50 + iqr_high * matched.spread_widening,
        })
    scenarios.sort(key=lambda s: s["key"])

    def blend(pick):
        """Geometric mean over the multi-thread scenarios: one ranking index
        that no single base can dominate."""
        logs = [math.log(pick(s)) for s in scenarios
                if not s["key"].endswith("_1t") and pick(s) > 0.0]
        return math.exp(sum(logs) / len(logs)) if logs else None

    return {
        "prediction_stage": matched.stage,
        "confidence": matched.confidence,
        "samples_used": len(matched.rows),
        "versions_used": versions_used,
        "scenarios": scenarios,
        "blended_rate_p25": blend(lambda s: s["rate_p25"]),
        "blended_rate_p50": blend(lambda s: s["rate_p50"]),
        "blended_rate_p75": blend(lambda s: s["rate_p75"]),
        "notes": matched.notes,
    }
