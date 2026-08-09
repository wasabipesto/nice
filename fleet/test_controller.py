"""Unit tests for the fleet controller's budget and economics logic.

Run: python3 -m unittest discover fleet
"""

import sqlite3
import time
import unittest

import controller
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
            "created_at, ttl_at, last_charged_at, destroyed_at, destroy_reason, spend) "
            "VALUES (?, 'x', 'exploit', 'RTX A4000', 0.04, ?, ?, ?, ?, 'preempted', ?)",
            (vast_id, now - 3600, now, now, now, spend),
        )

    def test_reconcile_trues_up_spend_and_bucket(self):
        db = memory_db()
        db.execute("INSERT INTO bucket (id, balance, updated_at) VALUES (1, 5.0, ?)",
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
        # Bucket charged the net drift: (0.05-0)+(0.06-0.10) = +0.01 -> 5.0-0.01.
        bal = db.execute("SELECT balance FROM bucket WHERE id = 1").fetchone()["balance"]
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


if __name__ == "__main__":
    unittest.main()
