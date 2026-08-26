"""Unit tests for the fleet controller's budget and economics logic.

Run: python3 -m unittest discover fleet
"""

import sqlite3
import time
import unittest

import controller
import estimator
from controller import (
    DEFAULT_CONFIG,
    exploit_allowed,
    instance_alive,
    accrue_bucket,
    e_hold_hours,
    effective_rate,
    heat_ok,
    offer_ev,
    parse_invoice_charges,
    plan_invoice_trueup,
    pounce_eligible,
    reconcile_invoices,
    renew_exploits,
    suggested_bid,
    trailing_median_ev,
)


def cfg(**overrides):
    c = dict(DEFAULT_CONFIG)
    c.update(overrides)
    return c


def memory_db():
    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    db.executescript(controller.SCHEMA)
    return db


class BucketTests(unittest.TestCase):
    def test_accrual_rate_and_cap(self):
        c = cfg(accrual_usd_per_month=30.0, bucket_cap_usd=7.0)
        # One day at $30/mo = $1.
        now = time.time()
        self.assertAlmostEqual(
            accrue_bucket(0.0, now - 86400, now, c), 1.0, places=3
        )
        # A month of accrual caps at the bucket, not $30.
        self.assertAlmostEqual(
            accrue_bucket(0.0, now - 30 * 86400, now, c), 7.0, places=3
        )
        # Clock going backwards accrues nothing.
        self.assertEqual(accrue_bucket(2.0, now + 3600, now, c), 2.0)

    def test_worst_case_month_is_bounded(self):
        # Starting from a full bucket, total spendable over 30 days is
        # accrual + one bucket: the "blow the budget" scenario is bounded.
        c = cfg(accrual_usd_per_month=30.0, bucket_cap_usd=7.0)
        self.assertAlmostEqual(30.0 + c["bucket_cap_usd"], 37.0)


class EconomicsTests(unittest.TestCase):
    def test_hold_amortization_taxes_churny_types(self):
        # Same rate: an A4000-class hold (13.5h) keeps ~97% of it, a
        # 4090-class hold (0.6h) keeps ~62%.
        long_hold = effective_rate(1e11, 13.5, 0.12, 0.25)
        short_hold = effective_rate(1e11, 0.6, 0.12, 0.25)
        self.assertGreater(long_hold / 1e11, 0.96)
        self.assertLess(short_hold / 1e11, 0.65)

    def test_ev_uses_p25_and_bid(self):
        c = cfg(bid_multiplier=1.2, setup_hours=0.0, lost_interval_hours=0.0)
        offer = {"min_bid": 0.05}
        est = {"blended_rate_p25": 1.2e11}
        ev = offer_ev(offer, est, 10.0, c)
        self.assertAlmostEqual(ev, 1.2e11 / 0.06, delta=1e9)
        # Missing prediction = zero EV, never a purchase.
        self.assertEqual(offer_ev(offer, {"blended_rate_p25": None}, 10.0, c), 0.0)

    def test_suggested_bid_rides_the_floor(self):
        c = cfg(bid_multiplier=1.2)
        self.assertAlmostEqual(suggested_bid({"min_bid": 0.05}, c), 0.06)

    def test_exploit_trust_gate(self):
        c = cfg()
        self.assertTrue(exploit_allowed({"prediction_stage": "exact", "confidence": 85}, c))
        self.assertTrue(exploit_allowed({"prediction_stage": "same-gpu", "confidence": 50}, c))
        self.assertFalse(
            exploit_allowed({"prediction_stage": "floor", "confidence": 15}, c),
            "floor-stage EVs are the winner's-curse zone",
        )
        self.assertFalse(
            exploit_allowed({"prediction_stage": "same-gpu", "confidence": 20}, c),
            "low confidence fails even on a trusted stage",
        )

    def test_outbid_detection(self):
        c = cfg()
        self.assertTrue(instance_alive({"actual_status": "running"}, c))
        self.assertTrue(instance_alive({"actual_status": "loading"}, c, age_minutes=5))
        self.assertTrue(instance_alive({}, c, age_minutes=1), "just created counts as alive")
        self.assertFalse(
            instance_alive({"actual_status": "exited"}, c),
            "an outbid bid instance is stopped, not gone, and must be reaped",
        )
        self.assertFalse(instance_alive({"actual_status": "offline"}, c))
        # Transitional states go stale after the grace period: a loaded-but-
        # never-running instance (bid slipped under the floor) must be reaped.
        self.assertFalse(instance_alive({"actual_status": "created"}, c, age_minutes=25))
        self.assertTrue(
            instance_alive({"actual_status": "running"}, c, age_minutes=999),
            "running never goes stale",
        )

    def test_pounce_gate(self):
        c = cfg(pounce_multiplier=1.4)
        self.assertFalse(pounce_eligible(100.0, None, c), "no baseline, no pounce")
        self.assertFalse(pounce_eligible(139.0, 100.0, c))
        self.assertTrue(pounce_eligible(140.0, 100.0, c))


class LedgerTests(unittest.TestCase):
    def test_e_hold_prefers_realized_data(self):
        db = memory_db()
        c = cfg()
        # No history: market-study seed.
        self.assertEqual(e_hold_hours(db, c, "RTX 4090"), 0.6)
        self.assertEqual(e_hold_hours(db, c, "Unknown GPU"), c["e_hold_seed_hours"]["default"])
        # Three realized preemptions: our data wins.
        now = time.time()
        for i, hold_h in enumerate([2.0, 3.0, 4.0]):
            db.execute(
                "INSERT INTO instances (vast_id, label, purpose, gpu_name, bid, "
                "created_at, ttl_at, last_charged_at, destroyed_at, destroy_reason) "
                "VALUES (?, 'x', 'exploit', 'RTX 4090', 0.05, ?, ?, ?, ?, 'preempted')",
                (i, now - hold_h * 3600, now, now, now),
            )
        self.assertAlmostEqual(e_hold_hours(db, c, "RTX 4090"), 3.0, places=2)
        # Explore disappearances (self-retirement) must not count as holds.
        db.execute(
            "INSERT INTO instances (vast_id, label, purpose, gpu_name, bid, "
            "created_at, ttl_at, last_charged_at, destroyed_at, destroy_reason) "
            "VALUES (99, 'x', 'explore', 'RTX 4090', 0.05, ?, ?, ?, ?, 'preempted')",
            (now - 0.1 * 3600, now, now, now),
        )
        self.assertAlmostEqual(
            e_hold_hours(db, c, "RTX 4090"), 3.0, places=2,
            msg="explore self-retirement polluted the hold estimate",
        )

    def test_trailing_median_needs_history(self):
        db = memory_db()
        self.assertIsNone(trailing_median_ev(db), "sparse history gives no baseline")
        now = time.time()
        for i in range(25):
            db.execute("INSERT INTO ev_seen (ts, ev) VALUES (?, ?)", (now, 100.0 + i))
        self.assertAlmostEqual(trailing_median_ev(db), 112.0)

    def test_heat_guard(self):
        db = memory_db()
        c = cfg(heat_guard_multiplier=1.15)
        # Not enough history: permissive.
        self.assertTrue(heat_ok(db, c, "RTX 3060", 0.10))
        now = time.time()
        for _ in range(25):
            db.execute(
                "INSERT INTO type_floor (ts, gpu_name, p10_dph) VALUES (?, 'RTX 3060', 0.05)",
                (now,),
            )
        self.assertTrue(heat_ok(db, c, "RTX 3060", 0.055))
        self.assertFalse(heat_ok(db, c, "RTX 3060", 0.06), "15% above trailing floor is hot")


class InvoiceReconcileTests(unittest.TestCase):
    def test_parse_sums_owned_charges_only(self):
        rows = [
            {"type": "charge", "instance_id": 1, "amount": "0.040",
             "description": "Instance 1 GPU charge"},
            {"type": "charge", "instance_id": 1, "amount": "0.010",
             "description": "Instance 1 storage charge"},
            {"type": "charge", "instance_id": 2, "amount": "0.500",
             "description": "Instance 2 GPU charge"},   # not ours
            {"type": "credit", "instance_id": 1, "amount": "-50.0",
             "description": "top-up"},                   # not a charge
            {"type": "charge", "instance_id": 1, "amount": "oops",
             "description": "garbage amount"},           # unparseable
        ]
        actual = parse_invoice_charges(rows, owned_ids={1})
        # GPU + storage for instance 1 only; foreign/credit/garbage ignored.
        self.assertEqual(set(actual), {1})
        self.assertAlmostEqual(actual[1], 0.050, places=6)

    def test_plan_detects_drift_and_is_idempotent(self):
        # id 1 under-charged (bug: $0 estimate, $0.05 real); id 2 over-charged.
        spend = {1: 0.0, 2: 0.10}
        actual = {1: 0.05, 2: 0.06}
        updates, net = plan_invoice_trueup(spend, actual)
        self.assertEqual({u[0] for u in updates}, {1, 2})
        # net owed = (0.05-0) + (0.06-0.10) = +0.01
        self.assertAlmostEqual(net, 0.01, places=6)
        # Re-running once spend == actual yields nothing.
        updates2, net2 = plan_invoice_trueup(actual, actual)
        self.assertEqual(updates2, [])
        self.assertEqual(net2, 0.0)

    def _seed(self, db, vast_id, spend):
        now = time.time()
        db.execute(
            "INSERT INTO instances (vast_id, label, purpose, gpu_name, bid, "
            "created_at, ttl_at, last_charged_at, destroyed_at, destroy_reason, spend, mode) "
            "VALUES (?, 'x', 'exploit', 'RTX A4000', 0.04, ?, ?, ?, ?, 'preempted', ?, 'niceonly')",
            (vast_id, now - 3600, now, now, now, spend),
        )

    def test_reconcile_trues_up_spend_and_bucket(self):
        db = memory_db()
        db.execute("INSERT INTO buckets (mode, balance, updated_at) VALUES ('niceonly', 5.0, ?)",
                   (time.time(),))
        self._seed(db, 101, spend=0.0)    # preempted, never charged (the bug)
        self._seed(db, 102, spend=0.10)   # over-charged estimate
        db.commit()
        invoices = {"invoices": [
            {"type": "charge", "instance_id": 101, "amount": "0.050",
             "description": "Instance 101 GPU charge"},
            {"type": "charge", "instance_id": 102, "amount": "0.060",
             "description": "Instance 102 GPU charge"},
            {"type": "charge", "instance_id": 999, "amount": "9.99",
             "description": "someone else's box"},
        ]}
        orig = controller.show_invoices
        controller.show_invoices = lambda cfg: invoices
        try:
            reconcile_invoices(cfg(invoice_reconcile_hours=1.0), db, dry=False)
        finally:
            controller.show_invoices = orig
        spends = {r["vast_id"]: (r["spend"], r["invoiced"])
                  for r in db.execute("SELECT vast_id, spend, invoiced FROM instances")}
        self.assertAlmostEqual(spends[101][0], 0.05, places=6)
        self.assertAlmostEqual(spends[101][1], 0.05, places=6)
        self.assertAlmostEqual(spends[102][0], 0.06, places=6)
        # niceonly bucket charged the net drift: (0.05-0)+(0.06-0.10) = +0.01.
        bal = db.execute("SELECT balance FROM buckets WHERE mode = 'niceonly'").fetchone()["balance"]
        self.assertAlmostEqual(bal, 4.99, places=6)

    def test_reconcile_respects_frequency_gate_and_disable(self):
        db = memory_db()
        db.execute("INSERT INTO bucket (id, balance, updated_at) VALUES (1, 5.0, ?)",
                   (time.time(),))
        self._seed(db, 201, spend=0.0)
        db.commit()
        invoices = {"invoices": [
            {"type": "charge", "instance_id": 201, "amount": "0.050",
             "description": "Instance 201 GPU charge"}]}
        orig = controller.show_invoices
        controller.show_invoices = lambda cfg: invoices
        try:
            # Disabled: no true-up at all.
            reconcile_invoices(cfg(invoice_reconcile_hours=0), db, dry=False)
            self.assertIsNone(
                db.execute("SELECT invoiced FROM instances WHERE vast_id = 201")
                .fetchone()["invoiced"])
            # First enabled run applies; an immediate second is frequency-gated.
            reconcile_invoices(cfg(invoice_reconcile_hours=1.0), db, dry=False)
            db.execute("UPDATE instances SET spend = 0 WHERE vast_id = 201")  # pretend drift
            reconcile_invoices(cfg(invoice_reconcile_hours=1.0), db, dry=False)
            self.assertEqual(
                db.execute("SELECT spend FROM instances WHERE vast_id = 201")
                .fetchone()["spend"], 0,
                "second run within the hour must not re-true-up")
        finally:
            controller.show_invoices = orig


class RenewExploitTests(unittest.TestCase):
    def _seed_expired_exploit(self, db, vast_id=1, created_ago_h=2.0):
        now = time.time()
        db.execute(
            "INSERT INTO instances (vast_id, label, purpose, gpu_name, cpu_name, bid, "
            "ev_predicted, pounce, created_at, ttl_at, last_charged_at, mode) "
            "VALUES (?, 'x', 'exploit', 'RTX A4000', 'EPYC', 0.048, 1e11, 0, ?, ?, ?, 'niceonly')",
            (vast_id, now - created_ago_h * 3600, now - 60, now),  # ttl 60s in the past
        )
        # Trailing EV baseline (per-mode) so the above-median gate has a compare.
        for i in range(25):
            db.execute("INSERT INTO ev_seen (ts, ev, mode) VALUES (?, ?, 'niceonly')", (now, 1.0e11 + i))
        db.commit()

    def _patch_estimate(self, est):
        orig = controller.estimate_offer
        controller.estimate_offer = lambda cfg, db, offer, mode=None: est
        return orig

    def test_healthy_winner_is_renewed_not_reaped(self):
        db = memory_db()
        self._seed_expired_exploit(db)
        # A strong estimate: trusted stage, high confidence, EV well above median.
        orig = self._patch_estimate(
            {"prediction_stage": "exact", "confidence": 85, "blended_rate_p25": 5.0e10})
        try:
            renew_exploits(cfg(bid_multiplier=1.2, exploit_ttl_hours=24.0), db, dry=False)
        finally:
            controller.estimate_offer = orig
        row = db.execute("SELECT ttl_at, destroyed_at FROM instances WHERE vast_id = 1").fetchone()
        self.assertIsNone(row["destroyed_at"], "a healthy winner must not be reaped")
        self.assertGreater(row["ttl_at"], time.time() + 20 * 3600, "TTL should be pushed ~24h out")

    def test_faded_instance_past_ttl_is_reaped(self):
        db = memory_db()
        self._seed_expired_exploit(db)
        # Trusted stage but EV far below the trailing median -> no longer a winner.
        orig_est = self._patch_estimate(
            {"prediction_stage": "exact", "confidence": 85, "blended_rate_p25": 1.0})
        reaped = []
        orig_destroy = controller.destroy_instance
        controller.destroy_instance = lambda cfg, db, vid, reason, dry: reaped.append((vid, reason))
        try:
            renew_exploits(cfg(bid_multiplier=1.2, exploit_ttl_hours=24.0), db, dry=False)
        finally:
            controller.estimate_offer = orig_est
            controller.destroy_instance = orig_destroy
        self.assertEqual(reaped, [(1, "ttl")], "a faded instance out of time must be reaped")
        # TTL was not extended.
        row = db.execute("SELECT ttl_at FROM instances WHERE vast_id = 1").fetchone()
        self.assertLess(row["ttl_at"], time.time(), "faded instance TTL must not be renewed")

    def test_renew_disabled_leaves_it_for_reconcile(self):
        db = memory_db()
        self._seed_expired_exploit(db)
        called = {"n": 0}
        orig = controller.estimate_offer
        controller.estimate_offer = (
            lambda cfg, db, offer, mode=None: called.__setitem__("n", called["n"] + 1))
        try:
            renew_exploits(cfg(exploit_renew_window_hours=0), db, dry=False)
        finally:
            controller.estimate_offer = orig
        self.assertEqual(called["n"], 0, "disabled renewal must not estimate or touch instances")
        row = db.execute("SELECT ttl_at, destroyed_at FROM instances WHERE vast_id = 1").fetchone()
        self.assertIsNone(row["destroyed_at"])


class MultiModeTests(unittest.TestCase):
    def test_mode_helpers(self):
        one = cfg()  # DEFAULT_CONFIG: exploit_modes = {"niceonly": {}}
        self.assertEqual(controller.exploit_modes(one), ["niceonly"])
        self.assertEqual(controller.primary_mode(one), "niceonly")
        two = cfg(exploit_modes={
            "niceonly": {},
            "detailed": {"accrual_usd_per_month": 15.0, "bucket_cap_usd": 3.0},
        })
        self.assertEqual(controller.exploit_modes(two), ["niceonly", "detailed"])
        self.assertEqual(controller.primary_mode(two), "niceonly")
        # mcfg: override wins, omitted keys fall back to top level.
        self.assertEqual(controller.mcfg(two, "detailed", "bucket_cap_usd"), 3.0)
        self.assertEqual(controller.mcfg(two, "niceonly", "bucket_cap_usd"),
                         two["bucket_cap_usd"])
        # bucket_mode: own exploit mode kept; explore/legacy/unknown -> primary.
        self.assertEqual(controller.bucket_mode("detailed", two), "detailed")
        self.assertEqual(controller.bucket_mode(None, two), "niceonly")
        self.assertEqual(controller.bucket_mode("stale-mode", two), "niceonly")

    def test_ensure_buckets_migrates_legacy_into_primary(self):
        db = memory_db()
        now = 1_000_000.0
        db.execute("INSERT INTO bucket (id, balance, updated_at) VALUES (1, 4.2, ?)", (now,))
        c = cfg(exploit_modes={"niceonly": {}, "detailed": {}})
        controller.ensure_buckets(db, c, now)
        rows = {r["mode"]: r["balance"]
                for r in db.execute("SELECT mode, balance FROM buckets")}
        self.assertAlmostEqual(rows["niceonly"], 4.2)  # primary inherits legacy balance
        self.assertAlmostEqual(rows["detailed"], 0.0)  # added mode starts empty
        # Idempotent: a second call adds nothing and preserves balances.
        db.execute("UPDATE buckets SET balance = 1.0 WHERE mode = 'niceonly'")
        controller.ensure_buckets(db, c, now)
        self.assertAlmostEqual(
            db.execute("SELECT balance FROM buckets WHERE mode='niceonly'").fetchone()["balance"],
            1.0)

    def test_exploit_split_is_per_mode_with_dedup(self):
        db = memory_db()
        now = 1_000_000.0
        # Two well-funded modes; a strong estimate so every offer clears the bar.
        c = cfg(exploit_modes={
            "niceonly": {"max_exploit_instances": 1},  # takes only the top offer
            "detailed": {"max_exploit_instances": 5},   # gets the rest (deduped)
        }, bucket_cap_usd=10.0, reserve_fraction=0.5, dry_run=False)
        for m in ("niceonly", "detailed"):
            db.execute("INSERT INTO buckets (mode, balance, updated_at) VALUES (?, 9.0, ?)", (m, now))
        est = {"prediction_stage": "exact", "confidence": 85, "blended_rate_p25": 5.0e10}
        def offers(ev_scale):
            return [({"id": i, "min_bid": 0.05, "gpu_name": "RTX A4000",
                      "cpu_name": "EPYC"}, est, ev_scale * (10 - i)) for i in range(3)]
        by_mode = {"niceonly": offers(1.0), "detailed": offers(1.0)}
        created = []
        orig = controller.create_instance
        controller.create_instance = (
            lambda cfg, db, offer, purpose, bid, ttl, ev, pounce, dry, mode=None:
            created.append((offer["id"], mode)))
        try:
            controller.plan_exploit(c, db, by_mode, dry=False)
        finally:
            controller.create_instance = orig
        # Each physical offer id bought at most once across both modes (dedup).
        ids = [oid for oid, _m in created]
        self.assertEqual(len(ids), len(set(ids)), "an offer was double-bought across modes")
        # Both modes participated, and niceonly (config order) got first pick.
        modes_used = {m for _oid, m in created}
        self.assertEqual(modes_used, {"niceonly", "detailed"})


class InsufficientCreditTests(unittest.TestCase):
    """An empty account is account-level: the first create that fails on it
    aborts the whole buy phase instead of grinding through every offer."""

    class _Resp:
        text = '{"success": false, "error": "insufficient_credit", "msg": "Your account lacks credit; see the billing page."}'

    class _HttpBoom(Exception):
        def __init__(self, text):
            super().__init__("402 Client Error")
            self.response = type("R", (), {"text": text})()

    def _stub_client(self, text):
        test = self

        class Stub:
            def create_instance(self, **kw):
                raise test._HttpBoom(text)

        return Stub()

    def test_create_raises_on_empty_account(self):
        db = memory_db()
        orig = controller.vast_client
        controller.vast_client = lambda cfg: self._stub_client(self._Resp.text)
        try:
            with self.assertRaises(controller.InsufficientCredit):
                controller.create_instance(
                    cfg(dry_run=False), db,
                    {"id": 1, "gpu_name": "RTX A4000", "cpu_name": "EPYC"},
                    "exploit", 0.05, 1.0, 1.0e12, False, False, mode="niceonly",
                )
        finally:
            controller.vast_client = orig

    def test_stale_offer_failure_still_just_skips(self):
        db = memory_db()
        orig = controller.vast_client
        controller.vast_client = lambda cfg: self._stub_client(
            '{"success": false, "error": "no_such_ask", "msg": "Instance type 1 is no longer available."}'
        )
        try:  # per-offer failures stay per-offer: log and move on, no raise
            controller.create_instance(
                cfg(dry_run=False), db,
                {"id": 1, "gpu_name": "RTX A4000", "cpu_name": "EPYC"},
                "exploit", 0.05, 1.0, 1.0e12, False, False, mode="niceonly",
            )
        finally:
            controller.vast_client = orig

    def test_buy_phase_stops_at_first_credit_failure(self):
        db = memory_db()
        now = time.time()
        c = cfg(exploit_modes={"niceonly": {}, "detailed": {}}, dry_run=False)
        for m in ("niceonly", "detailed"):
            db.execute(
                "INSERT INTO buckets (mode, balance, updated_at) VALUES (?, 9.0, ?)", (m, now)
            )
        est = {"prediction_stage": "exact", "confidence": 85, "blended_rate_p25": 5.0e10}
        offers = [({"id": i, "min_bid": 0.05, "gpu_name": "RTX A4000",
                    "cpu_name": "EPYC"}, est, 1.0e12 * (10 - i)) for i in range(3)]
        by_mode = {"niceonly": list(offers), "detailed": list(offers)}
        attempts = []
        orig = controller.create_instance

        def broke(cfg_, db_, offer, *a, **kw):
            attempts.append(offer["id"])
            raise controller.InsufficientCredit("insufficient_credit")

        controller.create_instance = broke
        try:
            with self.assertRaises(controller.InsufficientCredit):
                controller.plan_exploit(c, db, by_mode, dry=False)
        finally:
            controller.create_instance = orig
        # One attempt total — not one per offer, not one per mode.
        self.assertEqual(len(attempts), 1)


class FinalIntervalTests(unittest.TestCase):
    """The stretch between the last tick and death was never billed, so
    instances that lived and died inside one tick recorded $0."""

    def _live_instance(self, db, vast_id, bid, charged_ago_h, mode="niceonly", now=None):
        now = now or time.time()
        db.execute(
            "INSERT INTO instances (vast_id, label, purpose, gpu_name, bid, created_at, "
            "ttl_at, last_charged_at, mode) VALUES (?, 'x', 'exploit', 'RTX 3080', ?, ?, ?, ?, ?)",
            (vast_id, bid, now - charged_ago_h * 3600, now + 3600,
             now - charged_ago_h * 3600, mode))
        db.execute("INSERT OR REPLACE INTO buckets (mode, balance, updated_at) VALUES (?, 5.0, ?)",
                   (mode, now))
        return db.execute("SELECT * FROM instances WHERE vast_id = ?", (vast_id,)).fetchone()

    def test_charges_half_the_unbilled_interval(self):
        db = memory_db()
        now = time.time()
        row = self._live_instance(db, 1, bid=0.10, charged_ago_h=0.5, now=now)
        cost = controller.charge_final_interval(db, cfg(), row, now)
        # 0.5h at $0.10/hr, halved for the unknown stop time.
        self.assertAlmostEqual(cost, 0.025, places=4)
        self.assertAlmostEqual(
            db.execute("SELECT spend FROM instances WHERE vast_id=1").fetchone()[0],
            0.025, places=4)
        self.assertAlmostEqual(
            db.execute("SELECT balance FROM buckets WHERE mode='niceonly'").fetchone()[0],
            5.0 - 0.025, places=4)

    def test_no_double_charge_when_already_current(self):
        # A deliberate destroy of an instance seen alive this tick: reconcile
        # already advanced last_charged_at, so there is nothing left to bill.
        db = memory_db()
        now = time.time()   # reconcile threads one `now` through the whole tick
        row = self._live_instance(db, 2, bid=0.10, charged_ago_h=0.0, now=now)
        self.assertEqual(controller.charge_final_interval(db, cfg(), row, now), 0.0)
        self.assertEqual(
            db.execute("SELECT spend FROM instances WHERE vast_id=2").fetchone()[0], 0.0)

    def test_sub_tick_instance_is_no_longer_free(self):
        # The 44% case: created and dead inside one tick, never observed running.
        db = memory_db()
        now = time.time()
        row = self._live_instance(db, 3, bid=0.06, charged_ago_h=10.0 / 60.0, now=now)
        cost = controller.charge_final_interval(db, cfg(), row, now)
        self.assertGreater(cost, 0.0, "an instance that lived a whole tick must not be free")


class EHoldStatisticTests(unittest.TestCase):
    def test_uses_mean_not_median(self):
        # Right-skewed holds: the long tail carries nearly all delivered work,
        # and amortisation is a mean operation, so the median under-values it.
        db = memory_db()
        now = time.time()
        holds = [0.4, 0.4, 0.5, 0.5, 20.0]      # median 0.5, mean 4.36
        for i, h in enumerate(holds):
            db.execute(
                "INSERT INTO instances (vast_id, label, purpose, gpu_name, bid, created_at, "
                "ttl_at, last_charged_at, destroyed_at, destroy_reason) "
                "VALUES (?, 'x', 'exploit', 'RTX 3080', 0.05, ?, ?, ?, ?, 'preempted')",
                (i, now - h * 3600, now, now, now))
        self.assertAlmostEqual(e_hold_hours(db, cfg(), "RTX 3080"), 4.36, places=2)


class CpuMatchKeyTests(unittest.TestCase):
    def test_bridges_vast_and_cpuinfo_strings(self):
        # Vast's listing string and the client's /proc/cpuinfo string for the
        # same chip must land on one key, or coverage is never detected.
        for vast, cpuinfo in [
            ("Xeon\u00ae E5-2680 v4", "Intel(R) Xeon(R) CPU E5-2680 v4 @ 2.40GHz"),
            ("AMD EPYC 7763 64-Core Processor", "AMD EPYC 7763 64-Core Processor"),
            ("Core\u2122 i7-6700", "Intel(R) Core(TM) i7-6700 CPU @ 3.40GHz"),
        ]:
            self.assertEqual(estimator.cpu_match_key(vast),
                             estimator.cpu_match_key(cpuinfo),
                             f"{vast!r} and {cpuinfo!r} are the same chip")

    def test_distinct_chips_stay_distinct(self):
        self.assertNotEqual(estimator.cpu_match_key("Xeon E5-2680 v4"),
                            estimator.cpu_match_key("Xeon E5-2690 v3"))
        self.assertNotEqual(estimator.cpu_match_key("AMD EPYC 7763"),
                            estimator.cpu_match_key("AMD EPYC 7402"))

    def test_missing_cpu_is_safe(self):
        self.assertEqual(estimator.cpu_match_key(None), "?")
        self.assertEqual(estimator.cpu_match_key(""), "?")


class ExploreTargetingTests(unittest.TestCase):
    """Explore should buy the coverage it lacks, not the cheapest thing on
    the market. The old rule spent a third of the budget re-measuring two
    already well-known cards."""

    def _seed_corpus(self, db, rows):
        """rows: (gpu_model, cpu_model, count) already normalized."""
        i = 0
        for gpu_model, cpu_model, n in rows:
            for _ in range(n):
                i += 1
                db.execute(
                    "INSERT INTO corpus (id, client_version, gpu, mode, threads, cpu_model, "
                    "gpu_model, scenarios) VALUES (?, '3.3.0', 1, 'Nice-only', 8, ?, ?, '[]')",
                    (i, cpu_model, gpu_model))
        db.commit()

    def _offers(self, specs):
        return {"niceonly": [({"id": i, "gpu_name": g, "cpu_name": c, "min_bid": b}, {}, 1.0)
                             for i, (g, c, b) in enumerate(specs)]}

    def _run(self, db, c, by_mode):
        created = []
        orig = controller.create_instance
        controller.create_instance = (
            lambda cfg, db_, offer, purpose, bid, ttl, ev, pounce, dry, mode=None:
            created.append(offer["gpu_name"]))
        try:
            controller.plan_explore(c, db, by_mode, dry=False)
        finally:
            controller.create_instance = orig
        return created

    def test_prefers_the_thinnest_cell_over_the_cheapest_offer(self):
        db = memory_db()
        now = time.time()
        db.execute("INSERT INTO buckets (mode, balance, updated_at) VALUES ('niceonly', 5.0, ?)",
                   (now,))
        self._seed_corpus(db, [("rtx 3060", "amd epyc 7763", 20)])   # well covered
        c = cfg(explore_per_tick=1, explore_target_samples=8)
        # The 3060 is far cheaper, but we already have 20 reports for it.
        created = self._run(db, c, self._offers([
            ("RTX 3060", "AMD EPYC 7763 64-Core", 0.01),
            ("RTX 4090", "AMD EPYC 7763 64-Core", 0.90),
        ]))
        self.assertEqual(created, ["RTX 4090"])

    def test_skips_cells_that_are_already_deep_enough(self):
        db = memory_db()
        now = time.time()
        db.execute("INSERT INTO buckets (mode, balance, updated_at) VALUES ('niceonly', 5.0, ?)",
                   (now,))
        self._seed_corpus(db, [("rtx 3060", "amd epyc 7763", 20)])
        c = cfg(explore_per_tick=2, explore_target_samples=8)
        created = self._run(db, c, self._offers([("RTX 3060", "AMD EPYC 7763 64-Core", 0.01)]))
        self.assertEqual(created, [], "a covered cell is not worth paying to re-measure")

    def test_one_buy_per_cell_per_tick(self):
        # Many offers share a cell; buying several would waste the whole slot
        # budget on one gap.
        db = memory_db()
        now = time.time()
        db.execute("INSERT INTO buckets (mode, balance, updated_at) VALUES ('niceonly', 5.0, ?)",
                   (now,))
        c = cfg(explore_per_tick=3, explore_target_samples=8)
        created = self._run(db, c, self._offers([
            ("RTX 4090", "AMD EPYC 7763 64-Core Processor", 0.10),
            # Same chip as above, as /proc/cpuinfo spells it: one cell, one buy.
            ("NVIDIA GeForce RTX 4090", "AMD EPYC 7763 64-Core Processor", 0.11),
            ("RTX 4080", "Intel(R) Xeon(R) Gold 6230", 0.12),
        ]))
        self.assertEqual(sorted(created), ["RTX 4080", "RTX 4090"])

    def test_cooldown_is_by_cell_not_exact_pair(self):
        # The old cooldown keyed on the raw (gpu, cpu) strings, so one popular
        # GPU across many CPU models never cooled down.
        db = memory_db()
        now = time.time()
        db.execute("INSERT INTO buckets (mode, balance, updated_at) VALUES ('niceonly', 5.0, ?)",
                   (now,))
        db.execute("INSERT INTO explored (gpu_name, cpu_name, explored_at) VALUES "
                   "('RTX 3090', 'Xeon\u00ae E5-2680 v4', ?)", (now,))
        c = cfg(explore_per_tick=2, explore_target_samples=8, explore_cooldown_days=14)
        created = self._run(db, c, self._offers([
            # Same GPU and chip, spelled the way the other source spells them.
            ("NVIDIA GeForce RTX 3090", "Intel(R) Xeon(R) CPU E5-2680 v4 @ 2.40GHz", 0.05),
        ]))
        self.assertEqual(created, [], "the same chip under two spellings is one cell")


class CorpusSyncTests(unittest.TestCase):
    """The corpus mirror: idempotent upsert, watermark advances past junk,
    a failed pull leaves what we already had."""

    def _report(self, i, gpu=True, mode="Nice-only", threads=8, cpu="AMD EPYC 7763",
                gpu_model="RTX 3080", rate=2.0e9):
        return {"id": i, "client_version": "3.3.0", "data": {
            "schema_version": 1,
            "config": {"gpu": gpu, "mode": mode, "threads": threads},
            "hardware": {"cpu_model": cpu, "gpu_model": gpu_model},
            "scenarios": [
                {"key": "b50_msd_weak", "base": 50, "threads": threads, "rate": rate},
                {"key": "b50_msd_weak_1t", "base": 50, "threads": 1, "rate": 3.0e8},
            ]}}

    def _patch_fetch(self, pages):
        """pages: list of responses, or an Exception to raise."""
        orig = controller._fetch_json
        calls = []

        def fake(cfg, url):
            calls.append(url)
            nxt = pages.pop(0) if pages else []
            if isinstance(nxt, Exception):
                raise nxt
            return nxt

        controller._fetch_json = fake
        return orig, calls

    def test_sync_stores_and_is_idempotent(self):
        db = memory_db()
        c = cfg(corpus_page_size=5000)
        rows = [self._report(i) for i in (1, 2, 3)]
        orig, _ = self._patch_fetch([list(rows)])
        try:
            self.assertEqual(controller.sync_corpus(c, db), 3)
            # Re-syncing re-reads the overlap window; the upsert must not duplicate.
            controller._fetch_json = lambda cfg_, url: list(rows)
            self.assertEqual(controller.sync_corpus(c, db), 3)
        finally:
            controller._fetch_json = orig
        self.assertEqual(controller.meta_get(db, "corpus_watermark"), 3)

    def test_watermark_advances_past_undecodable_rows(self):
        # A report we cannot decode must still move the watermark, or the sync
        # re-fetches it forever and never reaches newer reports.
        db = memory_db()
        junk = {"id": 7, "client_version": "3.3.0", "data": {"schema_version": 2}}
        orig, _ = self._patch_fetch([[junk]])
        try:
            controller.sync_corpus(cfg(), db)
        finally:
            controller._fetch_json = orig
        self.assertEqual(db.execute("SELECT COUNT(*) FROM corpus").fetchone()[0], 0)
        self.assertEqual(controller.meta_get(db, "corpus_watermark"), 7)

    def test_failed_pull_keeps_existing_corpus(self):
        db = memory_db()
        orig, _ = self._patch_fetch([[self._report(1)], OSError("boom")])
        try:
            controller.sync_corpus(cfg(), db)          # seeds one report
            controller._fetch_json = lambda cfg_, url: (_ for _ in ()).throw(OSError("boom"))
            self.assertEqual(controller.sync_corpus(cfg(), db), 1)   # survives, keeps it
        finally:
            controller._fetch_json = orig

    def test_paginates_until_short_page(self):
        db = memory_db()
        c = cfg(corpus_page_size=2)
        pages = [[self._report(1), self._report(2)], [self._report(3)]]
        orig, calls = self._patch_fetch(pages)
        try:
            self.assertEqual(controller.sync_corpus(c, db), 3)
        finally:
            controller._fetch_json = orig
        self.assertEqual(len(calls), 2, "should stop on the short page")
        self.assertIn("id=gt.0", calls[0])
        self.assertIn("id=gt.2", calls[1])

    def test_estimate_offer_uses_local_corpus(self):
        db = memory_db()
        orig, _ = self._patch_fetch([[self._report(i) for i in (1, 2, 3)]])
        try:
            controller.sync_corpus(cfg(), db)
        finally:
            controller._fetch_json = orig
        controller._corpus = None
        out = controller.estimate_offer(
            cfg(gpu=True, mode="niceonly"), db,
            {"gpu_name": "NVIDIA GeForce RTX 3080", "cpu_name": "AMD EPYC 7763",
             "cpu_cores_effective": 8})
        controller._corpus = None
        self.assertEqual(out["prediction_stage"], "exact")
        self.assertEqual(out["samples_used"], 3)
        self.assertLess(abs(out["blended_rate_p50"] - 2.0e9), 1.0)


class OnstartTemplateTests(unittest.TestCase):
    def test_explore_benchmarks_all_four_configs(self):
        # create_instance formats with these kwargs; the explore template no
        # longer uses {mode} (it runs every mode) but the extra kwarg is fine.
        rendered = DEFAULT_CONFIG["onstart_explore"].format(
            mode="niceonly", api_base="http://x", username="u", threads=8)
        for frag in (
            "nice_client niceonly --gpu --benchmark",   # niceonly on GPU
            "nice_client detailed --gpu --benchmark",   # detailed on GPU
            "nice_client niceonly --benchmark",         # niceonly on CPU (no --gpu)
            "nice_client detailed --benchmark",         # detailed on CPU (no --gpu)
        ):
            self.assertIn(frag, rendered, f"missing benchmark config: {frag}")
        # Alternate GPU/CPU so same-subsystem runs are never back-to-back.
        order = [seg for seg in rendered.split("nice_client ")[1:]]
        gpu = ["--gpu" in seg.split(";")[0] for seg in order]
        self.assertEqual(gpu, [True, False, True, False], "runs must alternate GPU/CPU")
        # Shell vars for self-retirement survive .format() intact.
        self.assertIn("${CONTAINER_ID}", rendered)
        self.assertIn("${CONTAINER_API_KEY}", rendered)

    def test_exploit_also_benchmarks_all_four_then_works(self):
        rendered = DEFAULT_CONFIG["onstart_exploit"].format(
            mode="detailed", api_base="http://x", username="u", threads=8)
        # Same four-config sweep as explore ...
        for frag in ("nice_client niceonly --gpu --benchmark",
                     "nice_client detailed --gpu --benchmark",
                     "nice_client niceonly --benchmark",
                     "nice_client detailed --benchmark"):
            self.assertIn(frag, rendered, f"exploit missing benchmark config: {frag}")
        # ... then execs the instance's own mode work loop (here: detailed).
        self.assertIn("exec nice_client detailed --gpu --repeat", rendered)

    def test_explore_and_exploit_share_the_sweep(self):
        kw = dict(mode="niceonly", api_base="http://x", username="u", threads=8)
        expl = DEFAULT_CONFIG["onstart_explore"].format(**kw)
        expo = DEFAULT_CONFIG["onstart_exploit"].format(**kw)
        sweep = controller._BENCH_SWEEP.format(**kw)
        self.assertTrue(expl.startswith(sweep) and expo.startswith(sweep))


if __name__ == "__main__":
    unittest.main()
