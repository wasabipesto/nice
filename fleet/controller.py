#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["vastai-sdk"]
# ///
"""Nice fleet controller: budgeted explore/exploit over the Vast bid market.

One tick per cron invocation:

  1. kill-switch check, then reconcile (always first: orphaned instances are
     the #1 cost risk) — diff our ledger against live Vast instances,
     destroy anything expired or unknown, charge runtime spend to the bucket
  2. accrue budget into the token bucket (sustained rate, capped)
  3. search bid offers, estimate each via the API's POST /estimate
  4. explore slice: benchmark hardware the estimator can't price
     (low confidence / cold-start stages), rate-limited and cooled down
  5. exploit slice: rank offers by hold-amortized P25 EV, buy under the
     reserve rules; exceptional deals may dip into the reserve ("pounce")
     with a probe TTL until realized data confirms the estimate

State lives in a sqlite ledger next to the config. Dry-run mode (the
default) plans and prints everything but creates and destroys nothing.

Design notes live in scratchpad/2026-08-vast-fleet/PLAN.md; budget scheme:
token bucket, ~$30/mo accrual, ~$7 cap, half-full reserve line, pounce at
>=1.4x trailing median EV with a one-hour probe.
"""

import argparse
import json
import sqlite3
import statistics
import sys
import time
import urllib.request

# ---------------------------------------------------------------------------
# Config and state

# Benchmark every mode x hardware config (niceonly/detailed x GPU/CPU) so the
# estimator gets a datapoint for each. Order alternates GPU/CPU so no two
# same-subsystem runs are back-to-back, with a short settle between, to keep
# thermal throttling from skewing the later runs. ';' (not '&&') so one config
# failing (e.g. no GPU) can't block the rest — partial data still uploads.
# Shared by explore (which then self-retires) and exploit (which then runs its
# work loop), so every machine we pay for seeds both estimators for its hardware.
_BENCH_SWEEP = (
    "nice_client niceonly --gpu --benchmark --benchmark-upload --no-progress "
    "--api-base {api_base} --username {username} --threads {threads} ; sleep 5 ; "
    "nice_client detailed --benchmark --benchmark-upload --no-progress "
    "--api-base {api_base} --username {username} --threads {threads} ; sleep 5 ; "
    "nice_client detailed --gpu --benchmark --benchmark-upload --no-progress "
    "--api-base {api_base} --username {username} --threads {threads} ; sleep 5 ; "
    "nice_client niceonly --benchmark --benchmark-upload --no-progress "
    "--api-base {api_base} --username {username} --threads {threads}"
)

DEFAULT_CONFIG = {
    "dry_run": True,
    "api_base": "https://api.nicenumbers.net",
    "username": "wasabipesto-fleet",
    "user_agent": "nice-fleet-controller/1.0",
    "vast_api_key": None,
    "label_prefix": "nice-fleet",
    "db_path": "fleet.sqlite3",
    "kill_switch_path": "KILL",
    # --- budget (token bucket) ---
    "accrual_usd_per_month": 30.0,
    "bucket_cap_usd": 7.0,
    "reserve_fraction": 0.5,
    "pounce_multiplier": 1.4,
    "pounce_probe_hours": 1.0,
    # How often to pull Vast's actual per-instance invoices and true-up the
    # ledger + bucket to ground truth. <= 0 disables (estimate-only). Invoices
    # finalize at day boundaries, so this is a lagging correction, not a meter.
    "invoice_reconcile_hours": 1.0,
    # --- economics ---
    "bid_multiplier": 1.2,
    "setup_hours": 0.12,
    "lost_interval_hours": 0.25,
    "heat_guard_multiplier": 1.15,
    # Statuses that count as "making progress". Anything else (exited,
    # stopped, offline...) means the bid lost or the host died: on Vast an
    # outbid instance is STOPPED, not destroyed — it lingers accruing
    # storage cost and must be reaped as a preemption.
    "alive_statuses": ["running", "loading", "created", "connecting"],
    # A non-running instance older than this is stuck (bid slipped below the
    # floor between create and start, pull failure, dead host...) and is
    # reaped; the next tick simply buys elsewhere. Generous enough for slow
    # image pulls.
    "stuck_grace_minutes": 20,
    "e_hold_seed_hours": {
        "default": 3.0,
        "RTX 4090": 0.6,
        "RTX 3090": 1.4,
        "RTX 3060": 3.3,
        "RTX A4000": 13.5,
        "A100 SXM4": 7.7,
        "H100 SXM": 4.4,
        "L40S": 0.5,
    },
    # --- fleet shape ---
    "mode": "niceonly",  # legacy single-mode default / api_estimate fallback
    # Exploit tracks: one independent budget per mode. Each mode gets its own
    # token bucket (accrual/cap/reserve/pounce) and instance cap; explore
    # benchmarks all modes and is funded from the FIRST (primary) mode's bucket.
    # A mode's block overrides top-level budget/shape keys; anything it omits
    # inherits the top-level default. The default here — one mode inheriting
    # everything — is byte-identical to the pre-multi-mode controller. To add a
    # second track, give each mode explicit accrual_usd_per_month/bucket_cap_usd
    # that SUM to the intended total (they are independent buckets, not a split).
    "exploit_modes": {"niceonly": {}},
    "gpu": True,
    "max_exploit_instances": 2,
    "exploit_stages": ["exact", "same-gpu-similar-cpu", "same-gpu", "same-cpu-scaled"],
    "exploit_min_confidence": 40,
    "exploit_ttl_hours": 6.0,
    # Keep proven winners: an ordinary exploit within this window of its TTL is
    # re-estimated and, if it still clears the trust bar and holds above-median
    # EV, renewed in place instead of churned (re-buying re-pays the benchmark
    # warmup + setup and risks a worse offer). The lifetime cap forces periodic
    # re-evaluation via a fresh buy. Set the window to 0 to disable renewal.
    "exploit_renew_window_hours": 1.0,
    "exploit_max_lifetime_hours": 168.0,
    "image": "ghcr.io/wasabipesto/nice_client:latest-gpu",
    "disk_gb": 30,
    # Instances launch in ssh runtype: Vast runs its own init and executes
    # onstart_cmd in a shell, so the image ENTRYPOINT is bypassed and the
    # templates below can call nice_client (on PATH in the image) directly.
    # Exploit runs the full sweep (a broken GPU crashes the repeat below anyway
    # and gets reaped), then execs its own mode's work loop. exec so nice_client
    # replaces the shell — Vast injects env into PID 1 and the client reads it.
    "onstart_exploit": (
        _BENCH_SWEEP + " ; "
        "exec nice_client {mode} --gpu --repeat --telemetry --no-progress "
        "--api-base {api_base} --username {username} --threads {threads}"
    ),
    # Explore runs the sweep then retires itself using the instance-scoped API
    # key Vast injects; the controller's TTL is the backstop if that fails.
    "onstart_explore": (
        _BENCH_SWEEP + " ; "
        "curl -s -X DELETE -H \"Authorization: Bearer ${{CONTAINER_API_KEY}}\" "
        "-H \"Content-Type: application/json\" -d \"{{}}\" "
        "\"https://console.vast.ai/api/v0/instances/${{CONTAINER_ID}}/\" ; "
        "sleep infinity"
    ),
    # --- explore slice ---
    "explore_confidence_threshold": 80,
    "explore_stages": ["floor", "none", "gpu-family-scaled", "cpu-family-scaled"],
    "explore_per_tick": 2,
    "explore_max_concurrent": 3,
    "explore_per_day": 10,
    "explore_cooldown_days": 14,
    "explore_ttl_minutes": 30,
    # --- offer filters (vast search query) ---
    "offer_query": (
        "reliability>0.95 num_gpus=1 gpu_ram>=8 cuda_vers>=12.0 rentable=true "
        "inet_down>100 disk_space>=30 dph_total<0.40"
    ),
    "offer_type": "bid",
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS bucket (   -- legacy single bucket; migrated to buckets
    id INTEGER PRIMARY KEY CHECK (id = 1),
    balance REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS buckets (  -- one token bucket per exploit mode
    mode TEXT PRIMARY KEY,
    balance REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS instances (
    vast_id INTEGER PRIMARY KEY,
    label TEXT NOT NULL,
    purpose TEXT NOT NULL,             -- 'explore' | 'exploit'
    gpu_name TEXT,
    cpu_name TEXT,
    bid REAL NOT NULL,
    ev_predicted REAL,
    pounce INTEGER NOT NULL DEFAULT 0,
    confirmed INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    ttl_at REAL NOT NULL,
    last_charged_at REAL NOT NULL,
    destroyed_at REAL,
    destroy_reason TEXT,
    spend REAL NOT NULL DEFAULT 0,   -- charged to the bucket so far (estimate,
                                     -- trued-up to invoiced once billing posts)
    invoiced REAL,                   -- actual Vast charge (GPU+storage+net),
                                     -- NULL until the instance is invoiced
    mode TEXT                        -- exploit mode (which bucket it charges);
                                     -- NULL for explore (funded by primary mode)
);
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value REAL
);
CREATE TABLE IF NOT EXISTS explored (
    gpu_name TEXT NOT NULL,
    cpu_name TEXT NOT NULL,
    explored_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS ev_seen (
    ts REAL NOT NULL,
    ev REAL NOT NULL,
    mode TEXT                        -- exploit mode this EV was priced for
);
CREATE TABLE IF NOT EXISTS type_floor (
    ts REAL NOT NULL,
    gpu_name TEXT NOT NULL,
    p10_dph REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    ts REAL NOT NULL,
    kind TEXT NOT NULL,
    detail TEXT NOT NULL
);
"""


def load_config(path):
    cfg = dict(DEFAULT_CONFIG)
    with open(path, encoding="utf-8") as f:
        cfg.update(json.load(f))
    return cfg


def open_db(path):
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    db.executescript(SCHEMA)
    # Migrations: add columns older ledgers predate.
    icols = {r["name"] for r in db.execute("PRAGMA table_info(instances)").fetchall()}
    if "invoiced" not in icols:
        db.execute("ALTER TABLE instances ADD COLUMN invoiced REAL")
    if "mode" not in icols:
        db.execute("ALTER TABLE instances ADD COLUMN mode TEXT")
    ecols = {r["name"] for r in db.execute("PRAGMA table_info(ev_seen)").fetchall()}
    if "mode" not in ecols:
        db.execute("ALTER TABLE ev_seen ADD COLUMN mode TEXT")
    if db.execute("SELECT COUNT(*) FROM bucket").fetchone()[0] == 0:
        db.execute(
            "INSERT INTO bucket (id, balance, updated_at) VALUES (1, 0, ?)",
            (time.time(),),
        )
    db.commit()
    return db


def log_event(db, kind, detail):
    db.execute(
        "INSERT INTO events (ts, kind, detail) VALUES (?, ?, ?)",
        (time.time(), kind, detail),
    )
    print(f"[{kind}] {detail}")


# ---------------------------------------------------------------------------
# Exploit modes: each mode is an independent budget track (its own bucket).

def exploit_modes(cfg):
    """Ordered list of exploit modes. Backward-compat: falls back to the single
    legacy cfg['mode'] when exploit_modes isn't configured."""
    modes = cfg.get("exploit_modes")
    return list(modes) if modes else [cfg.get("mode", "niceonly")]


def primary_mode(cfg):
    """The first mode: funds/charges explore and seeds the legacy bucket balance."""
    return exploit_modes(cfg)[0]


def mcfg(cfg, mode, key):
    """Per-mode config value, falling back to the top-level default when the
    mode's override block omits it."""
    return ((cfg.get("exploit_modes") or {}).get(mode) or {}).get(key, cfg[key])


def bucket_mode(row_mode, cfg):
    """Which bucket funds/charges an instance: its own exploit mode, else the
    primary mode (explore instances and legacy NULL-mode rows)."""
    return row_mode if row_mode in exploit_modes(cfg) else primary_mode(cfg)


def ensure_buckets(db, cfg, now):
    """Create a per-mode bucket for each exploit mode; on first migration seed
    the primary mode from the legacy single-row `bucket`, others from zero."""
    have = {r["mode"] for r in db.execute("SELECT mode FROM buckets").fetchall()}
    fresh = not have
    legacy = db.execute("SELECT balance, updated_at FROM bucket WHERE id = 1").fetchone()
    for i, mode in enumerate(exploit_modes(cfg)):
        if mode in have:
            continue
        if i == 0 and fresh and legacy is not None:
            bal, upd = legacy["balance"], legacy["updated_at"]
        else:
            bal, upd = 0.0, now
        db.execute(
            "INSERT INTO buckets (mode, balance, updated_at) VALUES (?, ?, ?)",
            (mode, bal, upd),
        )


def bucket_balance(db, mode):
    row = db.execute("SELECT balance FROM buckets WHERE mode = ?", (mode,)).fetchone()
    return row["balance"] if row else 0.0


def charge_bucket(db, mode, amt):
    """Debit (amt>0) or credit (amt<0) a mode's bucket."""
    db.execute("UPDATE buckets SET balance = balance - ? WHERE mode = ?", (amt, mode))


def meta_get(db, key, default=None):
    row = db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else default


def meta_set(db, key, value):
    db.execute(
        "INSERT INTO meta (key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )


# ---------------------------------------------------------------------------
# Pure budget / economics logic (unit-tested)


def accrue_bucket(balance, updated_at, now, cfg):
    """Token bucket accrual: continuous drip at the sustained rate, capped."""
    hours = max(0.0, (now - updated_at) / 3600.0)
    rate = cfg["accrual_usd_per_month"] / (30.0 * 24.0)
    return min(balance + hours * rate, cfg["bucket_cap_usd"])


def parse_invoice_charges(invoices, owned_ids):
    """Sum actual charge amounts per instance we own, across every charge kind
    (GPU + storage + upload + download). Vast returns amounts as strings and
    mixes in credits and foreign instances; non-charge rows, unparseable
    amounts, and ids we don't own are ignored."""
    actual = {}
    for r in invoices:
        if r.get("type") != "charge":
            continue
        iid = r.get("instance_id")
        if iid not in owned_ids:
            continue
        try:
            amt = float(r.get("amount") or 0.0)
        except (TypeError, ValueError):
            continue
        actual[iid] = actual.get(iid, 0.0) + amt
    return actual


def plan_invoice_trueup(spend_by_id, actual_by_id, eps=1e-6):
    """Pure planner for the invoice true-up. Given each instance's charged-so-far
    `spend` and its actual invoiced total, return (updates, net) where updates is
    [(vast_id, new_spend, delta)] for instances whose charge drifted from actual
    and net is the total to subtract from the bucket (positive = we under-charged
    and owe more). Idempotent: once spend == actual, nothing is returned."""
    updates = []
    net = 0.0
    for iid, actual in actual_by_id.items():
        delta = actual - spend_by_id.get(iid, 0.0)
        if abs(delta) < eps:
            continue
        updates.append((iid, actual, delta))
        net += delta
    return updates, net


def effective_rate(p25_rate, e_hold_hours, setup_hours, lost_hours):
    """Hold-amortized throughput: the spin-up tax. Setup and the expected
    lost interval at preemption are amortized over the type's expected
    hold, so churn-prone types pay automatically."""
    if p25_rate is None or p25_rate <= 0 or e_hold_hours <= 0:
        return 0.0
    return p25_rate * e_hold_hours / (e_hold_hours + setup_hours + lost_hours)


def offer_ev(offer, estimate, e_hold_hours, cfg, mode=None):
    """Numbers per dollar, using the conservative P25 of the prediction."""
    bid = suggested_bid(offer, cfg, mode)
    if bid <= 0:
        return 0.0
    rate = effective_rate(
        estimate.get("blended_rate_p25"),
        e_hold_hours,
        cfg["setup_hours"],
        cfg["lost_interval_hours"],
    )
    return rate / bid


def suggested_bid(offer, cfg, mode=None):
    """Ride the floor: a small multiplier on the offer's minimum bid.
    Premium buys hours, not days (2026-07 market study), so protection
    comes from type selection and buffer depth, not big overbids."""
    mult = mcfg(cfg, mode, "bid_multiplier") if mode else cfg["bid_multiplier"]
    return round(float(offer.get("min_bid", offer.get("dph_total", 0))) * mult, 5)


def instance_alive(live_row, cfg, age_minutes=0.0):
    """Is this live-listed instance still doing useful work? Running always
    counts; transitional states count only within the stuck grace period."""
    status = live_row.get("actual_status") or "created"
    if status == "running":
        return True
    return status in cfg["alive_statuses"] and age_minutes < cfg["stuck_grace_minutes"]


def exploit_allowed(est, cfg, mode=None):
    """Exploit buys only trust estimates from the allowlisted prediction
    stages at sufficient confidence. Floor/none-stage hardware — where a
    single wild sample can produce an absurd EV with no spread to hedge it
    — is the explore slice's job, not a purchase signal."""
    stages = mcfg(cfg, mode, "exploit_stages") if mode else cfg["exploit_stages"]
    min_conf = mcfg(cfg, mode, "exploit_min_confidence") if mode else cfg["exploit_min_confidence"]
    return est.get("prediction_stage") in stages and est.get("confidence", 0) >= min_conf


def pounce_eligible(ev, trailing_median, cfg):
    """An exceptional deal may dip into the reserve only when it beats the
    trailing median EV by the configured multiple. With no history yet,
    nothing qualifies — the first days build the baseline."""
    if trailing_median is None or trailing_median <= 0:
        return False
    return ev >= trailing_median * cfg["pounce_multiplier"]


def e_hold_hours(db, cfg, gpu_name):
    """Expected hold for a GPU type: our own realized holds when we have
    them (ground truth), else the market-study seed table. Only exploit
    instances count — explore instances retire themselves after their
    benchmark, which reconcile can't distinguish from a preemption."""
    rows = db.execute(
        "SELECT (destroyed_at - created_at) / 3600.0 AS h FROM instances "
        "WHERE gpu_name = ? AND destroyed_at IS NOT NULL "
        "AND destroy_reason = 'preempted' AND purpose = 'exploit' "
        "ORDER BY destroyed_at DESC LIMIT 20",
        (gpu_name,),
    ).fetchall()
    if len(rows) >= 3:
        return statistics.median(r["h"] for r in rows)
    seeds = cfg["e_hold_seed_hours"]
    return seeds.get(gpu_name, seeds["default"])


def trailing_median_ev(db, days=3.0, mode=None):
    """Trailing median EV for the pounce baseline. EV scales are mode-specific
    (detailed is ~20x slower than niceonly), so a mode compares only to its own
    history. mode=None pools all rows (legacy / single-mode)."""
    cutoff = time.time() - days * 86400
    if mode is None:
        rows = db.execute("SELECT ev FROM ev_seen WHERE ts > ?", (cutoff,)).fetchall()
    else:
        rows = db.execute(
            "SELECT ev FROM ev_seen WHERE ts > ? AND mode = ?", (cutoff, mode)
        ).fetchall()
    values = [r["ev"] for r in rows if r["ev"] > 0]
    return statistics.median(values) if len(values) >= 20 else None


def heat_ok(db, cfg, gpu_name, current_p10):
    """Skip buys while a type runs hot versus its own trailing floor —
    demand spikes are when not to buy."""
    rows = db.execute(
        "SELECT p10_dph FROM type_floor WHERE gpu_name = ? AND ts > ?",
        (gpu_name, time.time() - 7 * 86400),
    ).fetchall()
    if len(rows) < 20:
        return True  # not enough history to judge
    trailing = statistics.median(r["p10_dph"] for r in rows)
    return current_p10 <= trailing * cfg["heat_guard_multiplier"]


# ---------------------------------------------------------------------------
# Vast SDK and API shims

_client = None


def vast_client(cfg):
    """One SDK client per process; the key falls back to the CLI's saved
    configuration when not set in the config. Imported lazily so the pure
    budget/economics logic (and its tests) need no dependencies."""
    global _client
    if _client is None:
        from vastai_sdk import VastAI

        _client = VastAI(api_key=cfg.get("vast_api_key") or None)
    return _client


class VastError(Exception):
    """A Vast operation failed; the message carries the full response."""


class InsufficientCredit(Exception):
    """The account is out of credit. This is account-level, not per-offer:
    once one create fails on it, every later create this tick fails the same
    way (an empty account once burned 72 straight ticks x ~17 offers of
    futile creates), so the buy phase aborts for the tick."""


def search_offers(cfg):
    return vast_client(cfg).search_offers(
        query=cfg["offer_query"], type=cfg["offer_type"], limit=200
    )


def show_instances(cfg):
    return vast_client(cfg).show_instances()


def show_invoices(cfg):
    """Actual per-instance billing rows: {invoices: [{type, description,
    timestamp, quantity, rate, amount, instance_id}, ...], current: {...}}."""
    return vast_client(cfg).show_invoices()


def reconcile_invoices(cfg, db, dry):
    """True-up ledger spend and the bucket from Vast's actual invoices, at most
    once per `invoice_reconcile_hours` (<= 0 disables).

    Vast bills the bid rate but only for GPU-running time, plus storage and
    network — none of which the per-tick bid*hold estimate captures, so left
    alone the bucket drifts (most visibly, it misses the runtime of instances
    preempted between ticks and records $0). This pulls the real charges and
    resets each instance's `spend` to actual, correcting the bucket by exactly
    the drift. Invoices finalize at day boundaries, so instances launched since
    the last close won't appear yet; those keep their conservative bid*hold
    estimate (an over-estimate — the safe direction for a budget gate) until a
    later pull trues them up."""
    every_h = cfg.get("invoice_reconcile_hours", 1.0)
    if not every_h or every_h <= 0:
        return
    now = time.time()
    if now - (meta_get(db, "last_invoice_reconcile_at", 0.0) or 0.0) < every_h * 3600.0:
        return
    try:
        resp = show_invoices(cfg)
    except Exception as e:
        log_event(db, "WARN", f"invoice pull failed, keeping estimates: {e!r}")
        return
    invoices = resp.get("invoices", []) if isinstance(resp, dict) else (resp or [])
    rows = db.execute("SELECT vast_id, spend, mode FROM instances").fetchall()
    owned = {r["vast_id"] for r in rows}
    mode_by_id = {r["vast_id"]: r["mode"] for r in rows}
    actual = parse_invoice_charges(invoices, owned)
    spend_by_id = {r["vast_id"]: r["spend"] for r in rows}
    updates, net = plan_invoice_trueup(spend_by_id, actual)
    total = sum(actual.values())
    # Route each instance's drift to the bucket that funds its mode.
    net_by_bucket = {}
    for iid, _new_spend, delta in updates:
        bm = bucket_mode(mode_by_id.get(iid), cfg)
        net_by_bucket[bm] = net_by_bucket.get(bm, 0.0) + delta
    if not dry:
        for iid, amt in actual.items():
            db.execute("UPDATE instances SET invoiced = ? WHERE vast_id = ?", (amt, iid))
        for iid, new_spend, _delta in updates:
            db.execute("UPDATE instances SET spend = ? WHERE vast_id = ?", (new_spend, iid))
        for bm, d in net_by_bucket.items():
            charge_bucket(db, bm, d)
        meta_set(db, "last_invoice_reconcile_at", now)
        db.commit()
    log_event(
        db, "INVOICE",
        f"trued-up {len(updates)}/{len(actual)} invoiced instances; "
        f"bucket adj ${-net:+.3f}; actual-to-date ${total:.2f}{' [DRY]' if dry else ''}",
    )


def create_instance(cfg, db, offer, purpose, bid, ttl_hours, ev, pounce, dry, mode=None):
    onstart_tpl = cfg["onstart_exploit"] if purpose == "exploit" else cfg["onstart_explore"]
    onstart = onstart_tpl.format(
        mode=mode or cfg.get("mode", "niceonly"),  # exploit runs this mode; explore ignores it
        api_base=cfg["api_base"],
        username=cfg["username"],
        threads=int(offer.get("cpu_cores_effective") or 4),
    )
    label = f"{cfg['label_prefix']}-{purpose}"
    desc = (
        f"{purpose} offer {offer['id']} {offer.get('gpu_name')} / "
        f"{(offer.get('cpu_name') or '?')[:40]} bid ${bid}/hr ev {ev:.3e}"
    )
    if dry:
        log_event(db, "DRY-CREATE", desc)
        return
    try:
        result = vast_client(cfg).create_instance(
            id=offer["id"],
            image=cfg["image"],
            disk=cfg["disk_gb"],
            label=label,
            onstart_cmd=onstart,
            price=bid,
            runtype="ssh",
            cancel_unavail=True,
        )
    except Exception as e:  # SDK raises assorted request errors; log fully
        body = getattr(getattr(e, "response", None), "text", "") or ""
        log_event(
            db, "ERROR",
            f"create failed for offer {offer['id']}: {e!r} {body[:200]}",
        )
        if "insufficient_credit" in body:
            raise InsufficientCredit(body[:200])
        return
    if not isinstance(result, dict) or not result.get("success"):
        log_event(db, "ERROR", f"create rejected for offer {offer['id']}: {result!r}")
        return
    vast_id = result.get("new_contract")
    now = time.time()
    db.execute(
        "INSERT INTO instances (vast_id, label, purpose, gpu_name, cpu_name, bid, "
        "ev_predicted, pounce, created_at, ttl_at, last_charged_at, mode) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            vast_id, label, purpose, offer.get("gpu_name"), offer.get("cpu_name"),
            bid, ev, int(pounce), now, now + ttl_hours * 3600, now, mode,
        ),
    )
    if purpose == "explore":
        db.execute(
            "INSERT INTO explored (gpu_name, cpu_name, explored_at) VALUES (?, ?, ?)",
            (offer.get("gpu_name") or "?", offer.get("cpu_name") or "?", now),
        )
    log_event(db, "CREATE", f"instance {vast_id}: {desc}")


def destroy_instance(cfg, db, vast_id, reason, dry):
    if dry:
        log_event(db, "DRY-DESTROY", f"instance {vast_id} ({reason})")
        return
    try:
        vast_client(cfg).destroy_instance(id=vast_id)
    except Exception as e:
        log_event(db, "ERROR", f"destroy {vast_id} failed: {e!r}")
        return
    db.execute(
        "UPDATE instances SET destroyed_at = ?, destroy_reason = ? WHERE vast_id = ?",
        (time.time(), reason, vast_id),
    )
    log_event(db, "DESTROY", f"instance {vast_id} ({reason})")


def api_estimate(cfg, offer, mode=None):
    body = {
        "mode": mode or cfg["mode"],
        "gpu": cfg["gpu"],
        "threads": int(offer.get("cpu_cores_effective") or 0) or None,
        "cpu_model": offer.get("cpu_name"),
        "gpu_model": offer.get("gpu_name"),
    }
    # A real User-Agent matters: the production API sits behind Cloudflare,
    # which rejects urllib's default Python-urllib UA with a 403.
    req = urllib.request.Request(
        f"{cfg['api_base']}/estimate",
        data=json.dumps(body).encode(),
        headers={
            "Content-Type": "application/json",
            "User-Agent": cfg["user_agent"],
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


# ---------------------------------------------------------------------------
# Tick phases


def reconcile(cfg, db, dry):
    """Diff ledger vs live instances. Runs first, unconditionally."""
    now = time.time()
    try:
        live = show_instances(cfg)
    except Exception as e:
        log_event(db, "ERROR", f"show instances failed, skipping tick: {e!r}")
        return None
    ours_live = {
        i["id"]: i for i in live if (i.get("label") or "").startswith(cfg["label_prefix"])
    }
    ledger = {
        r["vast_id"]: r
        for r in db.execute("SELECT * FROM instances WHERE destroyed_at IS NULL").fetchall()
    }

    # Live but unknown to the ledger: orphans. Destroy on sight.
    for vast_id in ours_live.keys() - ledger.keys():
        destroy_instance(cfg, db, vast_id, "orphan", dry)

    for vast_id, row in ledger.items():
        if vast_id not in ours_live:
            # Gone without us destroying it: preempted (or finished).
            hold_h = (now - row["created_at"]) / 3600.0
            db.execute(
                "UPDATE instances SET destroyed_at = ?, destroy_reason = 'preempted' "
                "WHERE vast_id = ?",
                (now, vast_id),
            )
            log_event(db, "PREEMPTED", f"instance {vast_id} after {hold_h:.2f}h")
            continue
        age_minutes = (now - row["created_at"]) / 60.0
        if not instance_alive(ours_live[vast_id], cfg, age_minutes):
            # Outbid or dead: the instance is stopped, not gone, and pays
            # storage until destroyed. Reap it and record the hold.
            hold_h = (now - row["created_at"]) / 3600.0
            destroy_instance(cfg, db, vast_id, "preempted", dry)
            log_event(db, "OUTBID", f"instance {vast_id} stopped after {hold_h:.2f}h; reaped")
            continue
        # Charge runtime spend since the last tick (GPU time only accrues
        # while running; a loading instance pays storage, which is noise at
        # our scale and reconciled against invoices).
        if ours_live[vast_id].get("actual_status") == "running":
            dt_h = (now - row["last_charged_at"]) / 3600.0
            cost = dt_h * row["bid"]
            db.execute(
                "UPDATE instances SET spend = spend + ?, last_charged_at = ? WHERE vast_id = ?",
                (cost, now, vast_id),
            )
            charge_bucket(db, bucket_mode(row["mode"], cfg), cost)
        else:
            db.execute(
                "UPDATE instances SET last_charged_at = ? WHERE vast_id = ?", (now, vast_id)
            )
        # Ordinary exploits get a renewal decision (renew_exploits) instead of
        # a blind TTL reap; explore and pounce instances still lapse here.
        ordinary_exploit = row["purpose"] == "exploit" and not row["pounce"]
        if now >= row["ttl_at"] and not (ordinary_exploit and cfg.get("exploit_renew_window_hours", 0) > 0):
            destroy_instance(cfg, db, vast_id, "ttl", dry)
    db.commit()
    return ours_live


def extend_or_probe_pounces(cfg, db, dry):
    """Pounce instances start on a short probe TTL. Extend only once the
    estimator — now fed by the instance's own uploaded benchmark — still
    supports the buy at a trustworthy stage."""
    now = time.time()
    rows = db.execute(
        "SELECT * FROM instances WHERE destroyed_at IS NULL AND pounce = 1 AND confirmed = 0"
    ).fetchall()
    for row in rows:
        m = bucket_mode(row["mode"], cfg)
        offer_like = {
            "cpu_name": row["cpu_name"],
            "gpu_name": row["gpu_name"],
            "min_bid": row["bid"] / mcfg(cfg, m, "bid_multiplier"),
        }
        try:
            est = api_estimate(cfg, offer_like, mode=m)
        except OSError as e:
            log_event(db, "WARN", f"estimate for pounce {row['vast_id']} failed: {e}")
            continue
        ev = offer_ev(offer_like, est, e_hold_hours(db, cfg, row["gpu_name"]), cfg, mode=m)
        trailing = trailing_median_ev(db, mode=m)
        good_stage = est.get("prediction_stage") in ("exact", "same-gpu-similar-cpu")
        if good_stage and pounce_eligible(ev, trailing, cfg):
            db.execute(
                "UPDATE instances SET confirmed = 1, ttl_at = ? WHERE vast_id = ?",
                (now + mcfg(cfg, m, "exploit_ttl_hours") * 3600, row["vast_id"]),
            )
            log_event(
                db, "POUNCE-CONFIRMED",
                f"instance {row['vast_id']} stage={est.get('prediction_stage')} ev={ev:.3e}",
            )
        elif now >= row["ttl_at"]:
            destroy_instance(cfg, db, row["vast_id"], "pounce-unconfirmed", dry)
    db.commit()


def renew_exploits(cfg, db, dry):
    """Keep proven winners. An ordinary exploit within its renew window of TTL
    is re-estimated; if it still clears the trust bar and holds above-median EV
    it is renewed in place, otherwise it is reaped now (reconcile deferred the
    ordinary-exploit TTL decision to here). A hard lifetime cap forces a fresh
    buy periodically. Pounces are handled by extend_or_probe_pounces."""
    window_h = cfg.get("exploit_renew_window_hours", 0)
    if not window_h or window_h <= 0:
        return
    now = time.time()
    window_end = now + window_h * 3600.0
    max_life = cfg.get("exploit_max_lifetime_hours", 168.0) * 3600.0
    rows = db.execute(
        "SELECT * FROM instances WHERE destroyed_at IS NULL AND purpose = 'exploit' "
        "AND pounce = 0 AND ttl_at <= ?",
        (window_end,),
    ).fetchall()
    for row in rows:
        expired = now >= row["ttl_at"]
        if now - row["created_at"] >= max_life:
            if expired:  # aged out: force re-evaluation via a fresh buy
                destroy_instance(cfg, db, row["vast_id"], "ttl-max-life", dry)
            continue
        m = bucket_mode(row["mode"], cfg)
        offer_like = {
            "cpu_name": row["cpu_name"],
            "gpu_name": row["gpu_name"],
            "min_bid": row["bid"] / mcfg(cfg, m, "bid_multiplier"),
        }
        try:
            est = api_estimate(cfg, offer_like, mode=m)
        except OSError as e:
            log_event(db, "WARN", f"renew estimate for {row['vast_id']} failed: {e}")
            continue  # can't judge; leave TTL as-is, re-check next tick
        ev = offer_ev(offer_like, est, e_hold_hours(db, cfg, row["gpu_name"]), cfg, mode=m)
        trailing = trailing_median_ev(db, mode=m)
        healthy = (
            exploit_allowed(est, cfg, mode=m)
            and ev > 0
            and (trailing is None or ev >= trailing)
        )
        if healthy:
            if not dry:
                db.execute(
                    "UPDATE instances SET ttl_at = ? WHERE vast_id = ?",
                    (now + mcfg(cfg, m, "exploit_ttl_hours") * 3600.0, row["vast_id"]),
                )
            log_event(
                db, "RENEW",
                f"instance {row['vast_id']} {row['gpu_name']} held; "
                f"stage={est.get('prediction_stage')} ev={ev:.3e}{' [DRY]' if dry else ''}",
            )
        elif expired:  # no longer a winner and out of time: reap
            destroy_instance(cfg, db, row["vast_id"], "ttl", dry)
    db.commit()


def record_type_floors(db, offers):
    """Persist per-type p10 price floors (mode-independent) for the heat guard,
    and keep the market tables bounded. Runs once per tick over the raw offers."""
    now = time.time()
    by_type = {}
    for offer in offers:
        by_type.setdefault(offer.get("gpu_name") or "?", []).append(
            float(offer.get("min_bid", offer.get("dph_total", 0)))
        )
    for gpu_name, prices in by_type.items():
        prices.sort()
        p10 = prices[max(0, int(len(prices) * 0.10) - 1)] if len(prices) > 1 else prices[0]
        db.execute(
            "INSERT INTO type_floor (ts, gpu_name, p10_dph) VALUES (?, ?, ?)",
            (now, gpu_name, p10),
        )
    cutoff = now - 30 * 86400
    db.execute("DELETE FROM ev_seen WHERE ts < ?", (cutoff,))
    db.execute("DELETE FROM type_floor WHERE ts < ?", (cutoff,))
    db.commit()


def record_ev_seen(db, mode, mode_offers):
    """Persist the EV distribution for one mode (its own pounce baseline)."""
    now = time.time()
    for _offer, _est, ev in mode_offers:
        if ev > 0:
            db.execute("INSERT INTO ev_seen (ts, ev, mode) VALUES (?, ?, ?)", (now, ev, mode))
    db.commit()


def plan_explore(cfg, db, by_mode, dry):
    """One explore instance benchmarks every mode, so a (gpu,cpu) pair is worth
    exploring if it's uncertain in ANY mode. Funded from the primary bucket."""
    now = time.time()
    balance = bucket_balance(db, primary_mode(cfg))
    if balance <= 0:
        log_event(db, "EXPLORE-SKIP", f"bucket at ${balance:.3f}; exploring costs money too")
        return
    today = now - 86400
    launched_today = db.execute(
        "SELECT COUNT(*) FROM instances WHERE purpose = 'explore' AND created_at > ?",
        (today,),
    ).fetchone()[0]
    active_explores = db.execute(
        "SELECT COUNT(*) FROM instances WHERE purpose = 'explore' AND destroyed_at IS NULL"
    ).fetchone()[0]
    slots = min(
        cfg["explore_per_tick"],
        max(0, cfg["explore_per_day"] - launched_today),
        max(0, cfg["explore_max_concurrent"] - active_explores),
    )
    if slots == 0:
        return
    # Per offer id: the estimates across every mode, plus the offer object.
    ests = {}
    for m, mode_offers in by_mode.items():
        for offer, est, _ev in mode_offers:
            ests.setdefault(offer["id"], (offer, []))[1].append(est)
    def uncertain(est):
        return (
            est.get("confidence", 0) < cfg["explore_confidence_threshold"]
            or est.get("prediction_stage") in cfg["explore_stages"]
        )
    cooldown = now - cfg["explore_cooldown_days"] * 86400
    for offer, mode_ests in sorted(ests.values(), key=lambda t: float(t[0].get("min_bid", 9e9))):
        if slots == 0:
            break
        if not any(uncertain(e) for e in mode_ests):
            continue
        seen = db.execute(
            "SELECT COUNT(*) FROM explored WHERE gpu_name = ? AND cpu_name = ? AND explored_at > ?",
            (offer.get("gpu_name") or "?", offer.get("cpu_name") or "?", cooldown),
        ).fetchone()[0]
        if seen:
            continue
        bid = suggested_bid(offer, cfg, primary_mode(cfg))
        create_instance(
            cfg, db, offer, "explore", bid,
            cfg["explore_ttl_minutes"] / 60.0, 0.0, pounce=False, dry=dry, mode=None,
        )
        slots -= 1
    db.commit()


def plan_exploit(cfg, db, by_mode, dry):
    """One independent exploit pass per mode, each against its own bucket, EV
    ranking, instance cap, reserve and pounce baseline. A physical offer can be
    bought for only one mode per tick, so passes share a `claimed` set; modes
    run in config order (earlier = first pick)."""
    # Active exploit instances tallied per funding bucket (legacy NULL -> primary).
    active_by_mode = {}
    for r in db.execute(
        "SELECT mode FROM instances WHERE destroyed_at IS NULL AND purpose = 'exploit'"
    ).fetchall():
        bm = bucket_mode(r["mode"], cfg)
        active_by_mode[bm] = active_by_mode.get(bm, 0) + 1

    claimed = set()
    for m in exploit_modes(cfg):
        mode_offers = by_mode.get(m, [])
        balance = bucket_balance(db, m)
        reserve = mcfg(cfg, m, "bucket_cap_usd") * mcfg(cfg, m, "reserve_fraction")
        cap = mcfg(cfg, m, "max_exploit_instances")
        active = active_by_mode.get(m, 0)
        trailing = trailing_median_ev(db, mode=m)

        ranked = sorted(
            (t for t in mode_offers if t[2] > 0 and exploit_allowed(t[1], cfg, mode=m)),
            key=lambda t: t[2],
            reverse=True,
        )
        skipped = sum(1 for t in mode_offers if t[2] > 0 and not exploit_allowed(t[1], cfg, mode=m))
        if skipped:
            log_event(db, "STAGE-SKIP", f"[{m}] {skipped} offers below exploit trust bar")
        for offer, est, ev in ranked:
            if active >= cap:
                break
            if offer["id"] in claimed:
                continue  # another mode already took this physical offer this tick
            gpu_name = offer.get("gpu_name") or "?"
            prices = sorted(
                float(o.get("min_bid", o.get("dph_total", 9e9)))
                for o, _e, _v in mode_offers
                if (o.get("gpu_name") or "?") == gpu_name
            )
            current_p10 = prices[max(0, int(len(prices) * 0.10) - 1)] if prices else 9e9
            if not heat_ok(db, cfg, gpu_name, current_p10):
                log_event(db, "HEAT-SKIP", f"[{m}] {gpu_name} running hot; skipping this tick")
                continue
            bid = suggested_bid(offer, cfg, m)
            ordinary = balance > reserve
            pounce = not ordinary and pounce_eligible(ev, trailing, cfg) and balance > 0
            if pounce and bid * mcfg(cfg, m, "pounce_probe_hours") > balance / 2:
                continue
            if not ordinary and not pounce:
                break  # bucket at/below reserve and nothing exceptional: hold
            ttl = mcfg(cfg, m, "pounce_probe_hours") if pounce else mcfg(cfg, m, "exploit_ttl_hours")
            create_instance(cfg, db, offer, "exploit", bid, ttl, ev, pounce, dry, mode=m)
            claimed.add(offer["id"])
            active += 1
            balance -= bid * ttl  # planning estimate only; real charge accrues per tick
    db.commit()


def confidence_spread(offers_by_est, cfg):
    """One-line summary of estimator confidence across this tick's offers, so
    the MARKET log shows how much the estimator has actually learned from the
    survey: median + p25/p75 spread + full range, and how many offers clear
    the exploit trust bar (confidence >= exploit_min_confidence)."""
    confs = sorted(float(est.get("confidence") or 0) for _, est, _ in offers_by_est)
    if not confs:
        return "confidence: n/a"
    pct = lambda q: confs[min(len(confs) - 1, max(0, int(len(confs) * q)))]
    bar = cfg["exploit_min_confidence"]
    n_bar = sum(1 for c in confs if c >= bar)
    return (
        f"confidence% p50={statistics.median(confs):.0f} "
        f"p25={pct(0.25):.0f} p75={pct(0.75):.0f} "
        f"range={confs[0]:.0f}-{confs[-1]:.0f}; "
        f"{n_bar}/{len(confs)} >= exploit bar ({bar})"
    )


def tick(cfg):
    dry = cfg["dry_run"]
    db = open_db(cfg["db_path"])
    now = time.time()
    ensure_buckets(db, cfg, now)  # create/migrate per-mode buckets
    db.commit()

    import os
    if os.path.exists(cfg["kill_switch_path"]):
        log_event(db, "KILL", "kill switch present: destroying everything, no purchases")
        live = reconcile(cfg, db, dry)
        for vast_id in (live or {}):
            destroy_instance(cfg, db, vast_id, "kill-switch", dry)
        db.commit()
        return

    # 1. Reconcile before anything else.
    live = reconcile(cfg, db, dry)
    if live is None:
        return  # Vast API unreachable; budget state untouched, try next tick

    # 1b. True-up spend + bucket from actual Vast invoices (hourly by default),
    # correcting the drift the per-tick bid*hold estimate leaves behind.
    reconcile_invoices(cfg, db, dry)

    # 1c. Renew proven winners before their TTL churns them.
    renew_exploits(cfg, db, dry)

    # 2. Accrue each mode's bucket.
    balances = {}
    for m in exploit_modes(cfg):
        row = db.execute("SELECT balance, updated_at FROM buckets WHERE mode = ?", (m,)).fetchone()
        cap = mcfg(cfg, m, "bucket_cap_usd")
        bal = accrue_bucket(row["balance"], row["updated_at"], now, {**cfg, "bucket_cap_usd": cap,
              "accrual_usd_per_month": mcfg(cfg, m, "accrual_usd_per_month")})
        db.execute("UPDATE buckets SET balance = ?, updated_at = ? WHERE mode = ?", (bal, now, m))
        balances[m] = bal
        log_event(db, "BUCKET", f"[{m}] balance ${bal:.3f} / cap ${cap:.2f}")
    db.commit()

    # 3. Pounce probes come before new purchases.
    extend_or_probe_pounces(cfg, db, dry)

    # 4. Market snapshot + estimates, priced for each exploit mode.
    try:
        offers = search_offers(cfg)
    except Exception as e:
        log_event(db, "ERROR", f"offer search failed: {e!r}")
        return
    modes = exploit_modes(cfg)
    by_mode = {m: [] for m in modes}
    for offer in offers:
        gpu = offer.get("gpu_name") or "?"
        hold = e_hold_hours(db, cfg, gpu)
        for m in modes:
            try:
                est = api_estimate(cfg, offer, mode=m)
            except OSError as e:
                log_event(db, "WARN", f"estimate [{m}] failed for offer {offer.get('id')}: {e}")
                continue
            by_mode[m].append((offer, est, offer_ev(offer, est, hold, cfg, mode=m)))
    record_type_floors(db, offers)
    for m in modes:
        record_ev_seen(db, m, by_mode[m])
        log_event(db, "MARKET", f"[{m}] {len(by_mode[m])} offers; {confidence_spread(by_mode[m], cfg)}")

    # 5. Explore (all modes at once), then exploit (per mode).
    try:
        plan_explore(cfg, db, by_mode, dry)
        plan_exploit(cfg, db, by_mode, dry)
    except InsufficientCredit:
        # Commit first: creates that succeeded earlier this tick are already
        # on Vast, and an uncommitted ledger row would get them reaped as
        # orphans next tick.
        db.commit()
        log_event(db, "CREDIT", "account out of credit; buys aborted for this tick")

    # Tick summary.
    active = db.execute(
        "SELECT COUNT(*), COALESCE(SUM(spend), 0) FROM instances WHERE destroyed_at IS NULL"
    ).fetchone()
    month_spend = db.execute(
        "SELECT COALESCE(SUM(spend), 0) FROM instances WHERE created_at > ?",
        (now - 30 * 86400,),
    ).fetchone()[0]
    buckets_str = " ".join(f"{m}=${balances[m]:.3f}" for m in modes)
    log_event(
        db, "SUMMARY",
        f"active={active[0]} active_spend=${active[1]:.3f} 30d_spend=${month_spend:.3f} "
        f"buckets[{buckets_str}]{' [DRY RUN]' if dry else ''}",
    )
    db.commit()


def main():
    parser = argparse.ArgumentParser(description="Nice fleet controller")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--live", action="store_true", help="override config dry_run")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.live:
        cfg["dry_run"] = False
    tick(cfg)


if __name__ == "__main__":
    sys.exit(main())
