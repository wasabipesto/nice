"""Unit tests for the local estimator.

Ported one-for-one from the #[cfg(test)] block in common/src/estimator.rs so
the two stay comparable: same fixtures, same assertions, same expected stage
names and confidence numbers. A drift in either implementation should fail
here.

Run: python3 -m unittest discover fleet
"""

import unittest

from estimator import (
    decode_sample,
    estimate,
    family_key,
    gpu_models_match,
    mode_string,
    normalize_gpu_model,
    normalize_model,
)
from estimator import Sample


def sample(gpu, cpu_model, gpu_model, threads, multi_rate, single_rate):
    return Sample(
        client_version="3.4.0",
        gpu=gpu,
        mode="Nice-only",
        threads=threads,
        cpu_model=normalize_model(cpu_model),
        gpu_model=normalize_gpu_model(gpu_model) if gpu_model else None,
        scenarios=[
            {"key": "b50_msd_weak", "base": 50, "threads": threads, "rate": multi_rate},
            {"key": "b50_msd_weak_1t", "base": 50, "threads": 1, "rate": single_rate},
        ],
    )


def inp(gpu, cpu=None, gpu_model=None, threads=None, **kw):
    body = {"gpu": gpu, "mode": "Nice-only", "threads": threads,
            "cpu_model": cpu, "gpu_model": gpu_model, "base": None, "client_version": None}
    body.update(kw)
    return body


def multi(out):
    return next(s for s in out["scenarios"] if s["key"] == "b50_msd_weak")


class NormalizationTests(unittest.TestCase):
    def test_gpu_names_match_across_vast_and_cuda(self):
        # Vast offer listings vs CUDA device names must land on the same key.
        for vast, cuda in [("RTX 3080", "NVIDIA GeForce RTX 3080"),
                           ("RTX 3060", "NVIDIA GeForce RTX 3060"),
                           ("RTX A4000", "NVIDIA RTX A4000"),
                           ("A100 PCIE", "NVIDIA A100-PCIE-40GB")]:
            self.assertTrue(
                gpu_models_match(normalize_gpu_model(vast), normalize_gpu_model(cuda)),
                f"{vast} should match {cuda}")
        # Distinct models must not collapse into each other.
        for a, b in [("RTX 3060", "RTX 3060 Ti"), ("RTX 4070", "RTX 4070 Ti"),
                     ("RTX 3080", "RTX 3090")]:
            self.assertFalse(
                gpu_models_match(normalize_gpu_model(a), normalize_gpu_model(b)),
                f"{a} must not match {b}")

    def test_model_normalization_and_family(self):
        self.assertEqual(normalize_model("AMD EPYC 7763 64-Core Processor"),
                         "amd epyc 7763 64-core processor")
        self.assertEqual(family_key(normalize_model("AMD EPYC 7763 64-Core Processor")),
                         "amd epyc")
        self.assertEqual(normalize_model("Intel(R) Xeon(R) Gold 6230"), "intel xeon gold 6230")

    def test_mode_strings(self):
        self.assertEqual(mode_string("niceonly"), "Nice-only")
        self.assertEqual(mode_string("Nice-Only"), "Nice-only")
        self.assertEqual(mode_string("detailed"), "Detailed")
        self.assertIsNone(mode_string("bogus"))


class CpuChainTests(unittest.TestCase):
    def test_exact_cpu_match(self):
        samples = [sample(False, "AMD EPYC 7763", None, 8, 2.0e9, 3.0e8)]
        out = estimate(samples, inp(False, cpu="AMD EPYC 7763", threads=8))
        self.assertEqual(out["prediction_stage"], "exact")
        self.assertEqual(out["confidence"], 85)
        self.assertLess(abs(multi(out)["rate_p50"] - 2.0e9), 1.0)

    def test_same_cpu_scales_by_threads_with_ceiling(self):
        # 8 threads measured at 2e9; asking for 16 doubles linearly (the
        # 3e8 single-thread x 16 = 4.8e9 ceiling doesn't bind).
        samples = [sample(False, "AMD EPYC 7763", None, 8, 2.0e9, 3.0e8)]
        out = estimate(samples, inp(False, cpu="AMD EPYC 7763", threads=16))
        self.assertEqual(out["prediction_stage"], "same-cpu-scaled")
        self.assertLess(abs(multi(out)["rate_p50"] - 4.0e9), 1.0)

        # With a low single-thread rate the ceiling binds: 1e8 x 64 = 6.4e9,
        # below the 16e9 linear projection.
        samples = [sample(False, "AMD EPYC 7763", None, 8, 2.0e9, 1.0e8)]
        out = estimate(samples, inp(False, cpu="AMD EPYC 7763", threads=64))
        self.assertLess(abs(multi(out)["rate_p50"] - 6.4e9), 1.0)
        self.assertLess(out["confidence"], 60, "extrapolation must discount confidence")

    def test_family_fallback_and_floor(self):
        samples = [sample(False, "AMD EPYC 7763", None, 8, 2.0e9, 3.0e8)]
        out = estimate(samples, inp(False, cpu="AMD EPYC 9654", threads=8))
        self.assertEqual(out["prediction_stage"], "cpu-family-scaled")
        out = estimate(samples, inp(False, cpu="Intel Xeon Gold 6230", threads=8))
        self.assertEqual(out["prediction_stage"], "floor")
        self.assertEqual(out["confidence"], 15)


class GpuChainTests(unittest.TestCase):
    def test_gpu_chain(self):
        samples = [
            sample(True, "AMD EPYC 7763", "RTX 3080", 8, 2.0e9, 3.0e8),
            sample(True, "Intel Xeon Gold 6230", "RTX 3080", 32, 3.0e9, 3.0e8),
        ]
        # Same GPU and same CPU: exact.
        out = estimate(samples, inp(True, cpu="AMD EPYC 7763", gpu_model="RTX 3080", threads=8))
        self.assertEqual(out["prediction_stage"], "exact")
        # Same GPU, unseen CPU, thread count outside the 0.5-2.0 window of
        # both samples: falls to same-gpu.
        out = estimate(samples, inp(True, cpu="Unknown CPU", gpu_model="RTX 3080", threads=128))
        self.assertEqual(out["prediction_stage"], "same-gpu")
        self.assertTrue(any("MSD" in n for n in out["notes"]))
        # Unseen GPU: floor.
        out = estimate(samples, inp(True, cpu="AMD EPYC 7763", gpu_model="RTX 9999", threads=8))
        self.assertEqual(out["prediction_stage"], "floor")
        # Nothing in this mode/device class at all.
        out = estimate([], inp(True, cpu="AMD EPYC 7763", gpu_model="RTX 3080", threads=8))
        self.assertEqual(out["prediction_stage"], "none")
        self.assertEqual(out["confidence"], 0)

    def test_similar_cpu_stage_uses_thread_ratio(self):
        samples = [sample(True, "AMD EPYC 7763", "RTX 3080", 8, 2.0e9, 3.0e8)]
        # 8 measured vs 16 requested: ratio 0.5, inside the window.
        out = estimate(samples, inp(True, cpu="Unknown CPU", gpu_model="RTX 3080", threads=16))
        self.assertEqual(out["prediction_stage"], "same-gpu-similar-cpu")
        self.assertEqual(out["confidence"], 65)


class VersionAndDecodeTests(unittest.TestCase):
    def test_version_mismatch_discounts(self):
        samples = [sample(False, "AMD EPYC 7763", None, 8, 2.0e9, 3.0e8)]
        out = estimate(samples, inp(False, cpu="AMD EPYC 7763", threads=8,
                                    client_version="9.9.9"))
        self.assertEqual(out["prediction_stage"], "exact")
        self.assertEqual(out["confidence"], 70)
        self.assertTrue(any("9.9.9" in n for n in out["notes"]))

    def test_decode_rejects_bad_schema(self):
        self.assertIsNone(decode_sample("x", {"schema_version": 2}))

    def test_decode_extracts_and_normalizes(self):
        s = decode_sample("3.4.0", {
            "schema_version": 1,
            "config": {"gpu": False, "mode": "Nice-only", "threads": 8},
            "hardware": {"cpu_model": "AMD EPYC 7763", "gpu_model": None},
            "scenarios": [
                {"key": "b50_msd_weak", "base": 50, "threads": 8, "rate": 2.0e9},
                {"key": "dropped", "base": 50, "threads": 8, "rate": 0},
            ],
        })
        self.assertIsNotNone(s)
        self.assertEqual(len(s.scenarios), 1, "zero-rate scenarios are dropped")
        self.assertEqual(s.cpu_model, "amd epyc 7763")
        self.assertIsNone(s.gpu_model)

    def test_decode_excludes_browser_reports(self):
        # The browser suite uploads with config.platform = "browser"; those
        # reports are stored but must never enter the native corpus (wasm
        # rates with no cpu_model would pool into the fallback buckets).
        report = {
            "schema_version": 1,
            "config": {"gpu": False, "mode": "Detailed", "threads": 8,
                       "platform": "browser"},
            "hardware": {"user_agent": "Mozilla/5.0"},
            "scenarios": [
                {"key": "b40_detailed", "base": 40, "threads": 8, "rate": 1.5e7},
            ],
        }
        self.assertIsNone(decode_sample("3.4.0-wasm-worker", report))
        # The same report without the marker decodes, so the exclusion is
        # the platform key and nothing else.
        native = {**report, "config": {k: v for k, v in report["config"].items()
                                       if k != "platform"}}
        self.assertIsNotNone(decode_sample("3.4.0", native))

    def test_decode_survives_malformed_reports(self):
        # A bad report must be skipped, never raise: the corpus is uploaded
        # by clients we don't control.
        for bad in [
            {"schema_version": 1, "config": {"gpu": False, "mode": "Nice-only",
                                             "threads": "eight"},
             "hardware": {}, "scenarios": [{"key": "k", "base": 50, "threads": 8, "rate": 1.0}]},
            {"schema_version": 1, "config": {"gpu": False, "mode": "Nice-only", "threads": 8},
             "hardware": {}, "scenarios": {"not": "a list"}},
            {"schema_version": 1, "config": {"gpu": False, "mode": "Nice-only", "threads": 8},
             "hardware": {}},
            {"schema_version": 1, "config": {"gpu": False, "mode": "Nice-only", "threads": 8},
             "hardware": {}, "scenarios": [{"key": "k", "base": 50, "threads": 8,
                                            "rate": "fast"}]},
            "not a dict",
        ]:
            self.assertIsNone(decode_sample("x", bad), f"should reject {bad!r}")


class BlendTests(unittest.TestCase):
    def test_single_thread_scenarios_excluded_from_blend(self):
        # The blended index is the geometric mean over multi-thread scenarios
        # only; the _1t anchor must not drag it down.
        samples = [sample(False, "AMD EPYC 7763", None, 8, 2.0e9, 3.0e8)]
        out = estimate(samples, inp(False, cpu="AMD EPYC 7763", threads=8))
        self.assertLess(abs(out["blended_rate_p50"] - 2.0e9), 1.0)

    def test_base_filter_restricts_scenarios(self):
        s = sample(False, "AMD EPYC 7763", None, 8, 2.0e9, 3.0e8)
        s.scenarios.append({"key": "b40_msd_strong", "base": 40, "threads": 8, "rate": 5.0e9})
        out = estimate([s], inp(False, cpu="AMD EPYC 7763", threads=8, base=40))
        self.assertEqual([x["key"] for x in out["scenarios"]], ["b40_msd_strong"])
        self.assertLess(abs(out["blended_rate_p50"] - 5.0e9), 1.0)


if __name__ == "__main__":
    unittest.main()
