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
  3. search bid offers and estimate each locally, against a corpus of
     benchmark reports mirrored from the API's database
  4. explore slice: benchmark the hardware the estimator can't price yet,
     ranked by how thin that GPU/CPU-family cell is, rate-limited and cooled
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
import fcntl
import json
import os
import sqlite3
import statistics
import sys
import time
import traceback
import urllib.request

import estimator

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
    # PostgREST host serving the benchmark corpus. Estimation is local, so the
    # controller mirrors `benchmarks` rather than calling POST /estimate per
    # offer; this is the only place it is read from.
    "data_base": "https://data.nicenumbers.net",
    "corpus_overlap": 50,
    "corpus_page_size": 5000,
    "corpus_timeout_secs": 120,
    "username": "wasabipesto-fleet",
    "user_agent": "nice-fleet-controller/1.0",
    "vast_api_key": None,
    "label_prefix": "nice-fleet",
    "db_path": "fleet.sqlite3",
    "kill_switch_path": "KILL",
    # Relative paths here resolve against the config file's own directory, so
    # cron needs only one absolute path on its command line.
    "tick_lock_path": ".tick.lock",
    "log_path": None,          # None = write to stdout, for interactive runs
    "log_max_bytes": 5242880,
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
    # A single pull that would cut recorded spend by more than this fraction of
    # the total is refused outright and logged. A true-up corrects drift; a mass
    # reduction means the invoice feed changed shape, which is a thing to look
    # at rather than to act on — the original bug wrote off 2372 GPU-hours.
    # Whether an invoice pull may actually change the ledger. False = observe
    # only: pull, log the shape, report what would have moved, change nothing.
    "invoice_apply_trueup": True,
    "invoice_max_drop_fraction": 0.25,
    # Log (never act on) a thin match that disagrees with its wider pool by more
    # than this factor. Diagnostics only: overriding thin matches was measured
    # and made predictions worse.
    "divergence_warn_factor": 2.0,
    "divergence_warn_below_samples": 5,
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
    "mode": "niceonly",  # legacy single-mode default / estimate_offer fallback
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
    # A coverage cell (GPU model x CPU family) at or above this many benchmark
    # reports is considered priceable and no longer worth exploring. Replaces
    # the old confidence/stage thresholds: with the corpus local, depth is
    # measurable directly rather than inferred from an estimate's self-report.
    "explore_target_samples": 8,
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
    ever_ran INTEGER NOT NULL DEFAULT 0,  -- seen `running` at least once, so a
                                     -- $0 invoice is a fact, not an unsettled bill
    invoiced REAL,                   -- actual Vast charge (GPU+storage+net),
                                     -- NULL until the instance is invoiced
    mode TEXT                        -- exploit mode (which bucket it charges);
                                     -- NULL for explore (funded by primary mode)
);
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value REAL
);
CREATE TABLE IF NOT EXISTS corpus (   -- local mirror of the API's benchmarks table
    id INTEGER PRIMARY KEY,           -- benchmarks.id upstream; the sync watermark
    client_version TEXT NOT NULL,
    gpu INTEGER NOT NULL,
    mode TEXT NOT NULL,
    threads INTEGER NOT NULL,
    cpu_model TEXT,                   -- normalized at sync time
    gpu_model TEXT,                   -- normalized at sync time
    scenarios TEXT NOT NULL           -- JSON: [{key, base, threads, rate}, ...]
);
CREATE INDEX IF NOT EXISTS idx_corpus_pool ON corpus (gpu, mode);
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
    """Load the config and resolve its relative paths against its own location.

    Relative to the config file, not the working directory: cron invokes this
    from an arbitrary cwd, and resolving against the config means one absolute
    path on the crontab line is enough to locate everything else."""
    cfg = dict(DEFAULT_CONFIG)
    with open(path, encoding="utf-8") as f:
        cfg.update(json.load(f))
    base = os.path.dirname(os.path.abspath(path))
    for key in ("db_path", "kill_switch_path", "log_path"):
        value = cfg.get(key)
        if value and not os.path.isabs(value):
            cfg[key] = os.path.join(base, value)
    return cfg


def acquire_tick_lock(path):
    """Take the single-tick lock, or return None if a tick is already running.

    Ticks can outlast the cron cadence, and two of them racing corrupts the
    ledger — reconcile crashed on sqlite's write lock 14 times before this
    existed. Skipping is right rather than queueing: the next scheduled tick
    picks up whatever this one would have done. The descriptor is deliberately
    leaked, so the lock lives exactly as long as the process."""
    fd = os.open(path, os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        return None
    return fd


def redirect_output_to_log(path, max_bytes):
    """Send this tick's output to the log file, rotating it first if oversized.

    Takes over the real file descriptors rather than just `sys.stdout`, so an
    uncaught traceback lands in the log with everything else instead of in
    cron's mail. Rotation happens here, before the file is opened, because a
    process holding the old descriptor would otherwise keep writing to the
    rotated-away inode."""
    if max_bytes and os.path.exists(path) and os.path.getsize(path) > max_bytes:
        os.replace(path, path + ".1")
    handle = open(path, "a", buffering=1, encoding="utf-8", errors="replace")
    os.dup2(handle.fileno(), sys.stdout.fileno())
    os.dup2(handle.fileno(), sys.stderr.fileno())
    return handle


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
    if "ever_ran" not in icols:
        db.execute("ALTER TABLE instances ADD COLUMN ever_ran INTEGER NOT NULL DEFAULT 0")
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


def plan_invoice_trueup(spend_by_id, actual_by_id, eps=1e-6, alive_ids=frozenset(),
                        max_drop_fraction=None):
    """Pure planner for the invoice true-up. Given each instance's charged-so-far
    `spend` and its actual invoiced total, return (updates, net, refusals) where
    updates is [(vast_id, new_spend, delta)] for instances whose charge drifted
    from actual and net is the total to subtract from the bucket (positive = we
    under-charged and owe more). Idempotent: once spend == actual, nothing is
    returned.

    Three guards, all learned from the true-up that silently zeroed 1465
    instances and 2372 GPU-hours of real spend:

    * A live instance is never reduced. Its bill is still accruing, so a
      partial invoice always looks like an over-charge.
    * Spend never goes negative, whatever the invoice claims.
    * If the pull would cut recorded spend by more than `max_drop_fraction` of
      the total, the whole batch is refused rather than applied. A true-up is a
      correction; a mass reduction means the invoice feed changed shape under
      us, and that is a thing to look at, not to act on.
    """
    updates = []
    net = 0.0
    refusals = []
    for iid, actual in actual_by_id.items():
        current = spend_by_id.get(iid, 0.0)
        delta = actual - current
        if abs(delta) < eps:
            continue
        if delta < 0 and iid in alive_ids:
            refusals.append((iid, "still running; bill not final"))
            continue
        new_spend = max(0.0, actual)
        delta = new_spend - current
        if abs(delta) < eps:
            continue
        updates.append((iid, new_spend, delta))
        net += delta
    if max_drop_fraction is not None:
        total = sum(spend_by_id.values())
        drop = -net
        if total > 0 and drop > total * max_drop_fraction:
            return [], 0.0, [(None, f"batch would cut recorded spend by "
                                    f"${drop:.2f} of ${total:.2f} "
                                    f"(> {max_drop_fraction:.0%}); refusing")]
    return updates, net, refusals


def log_invoice_shape(db, invoices, spend_by_id, owned):
    """Record what an invoice pull actually contained, without acting on it.

    Per-instance charges live in a ~2-day sliding window and each instance is
    only ever seen once, so the conditions governing when a zero is trustworthy
    have to be written against observed shapes rather than guessed. This logs
    those shapes so a week of them accumulates before any rule is written."""
    from collections import Counter
    days = Counter()
    kinds = Counter()
    zero_rows = 0
    per_instance = {}
    for r in invoices:
        if r.get("type") != "charge":
            continue
        ts = r.get("timestamp")
        day = time.strftime("%Y-%m-%d", time.gmtime(ts)) if ts else "?"
        days[day] += 1
        try:
            amt = float(r.get("amount") or 0.0)
        except (TypeError, ValueError):
            amt = 0.0
        try:
            qty = float(r.get("quantity") or 0.0)
        except (TypeError, ValueError):
            qty = 0.0
        if qty == 0.0 or amt == 0.0:
            zero_rows += 1
        desc = (r.get("description") or "").split(":")[0]
        kinds[desc.split()[-1] if desc.split() else "?"] += 1
        iid = r.get("instance_id")
        if iid in owned:
            per_instance[iid] = per_instance.get(iid, 0.0) + amt
    nonzero = {i: a for i, a in per_instance.items() if a > 0}
    ratios = [nonzero[i] / spend_by_id[i] for i in nonzero
              if spend_by_id.get(i, 0.0) > 0]
    ratio_note = (f"invoiced/estimated median {statistics.median(ratios):.2f} "
                  f"over {len(ratios)}") if ratios else "no comparable pairs"
    log_event(
        db, "INVOICE-SHAPE",
        f"days={dict(days)} kinds={dict(kinds)} zero_rows={zero_rows} "
        f"ours={len(per_instance)} of_which_nonzero={len(nonzero)}; {ratio_note}",
    )


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
    benchmark, which reconcile can't distinguish from a preemption.

    The MEAN, not the median. `effective_rate` amortizes a fixed setup cost
    over this, and across many instances the true ratio is
    sum(hold) / sum(hold + setup + lost) — a mean operation. Hold
    distributions are violently right-skewed (RTX 3080: median 0.97h, mean
    4.63h), and the long tail is where nearly all delivered work lives, so
    the median charged a 72% amortization haircut where the truth was ~7%
    and systematically undervalued any type that usually dies young but
    occasionally holds for a day.

    Known residual bias: sampling the most recent N *deaths* under-represents
    long holds, which die less often per unit time. Correcting that properly
    is survival analysis; the window is widened instead, which trades some
    responsiveness for stability.
    """
    rows = db.execute(
        "SELECT (destroyed_at - created_at) / 3600.0 AS h FROM instances "
        "WHERE gpu_name = ? AND destroyed_at IS NOT NULL "
        "AND destroy_reason = 'preempted' AND purpose = 'exploit' "
        "ORDER BY destroyed_at DESC LIMIT 40",
        (gpu_name,),
    ).fetchall()
    if len(rows) >= 3:
        return statistics.mean(r["h"] for r in rows)
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
    # Pulling and applying are separate switches. The window holds ~2 days and
    # each instance appears in it once, so the observations needed to write the
    # true-up's conditions have to be collected while the true-up itself stays
    # off; `invoice_apply_trueup: false` is that observation mode.
    apply_trueup = cfg.get("invoice_apply_trueup", True)
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
    # Always record the pull's shape, even when the true-up itself is disabled:
    # the conditions governing when a zero charge is trustworthy have to be
    # written against observed data, and the window only holds ~2 days.
    log_invoice_shape(db, invoices, spend_by_id, owned)
    alive_ids = {
        r["vast_id"] for r in db.execute(
            "SELECT vast_id FROM instances WHERE destroyed_at IS NULL").fetchall()
    }
    updates, net, refusals = plan_invoice_trueup(
        spend_by_id, actual, alive_ids=alive_ids,
        max_drop_fraction=cfg.get("invoice_max_drop_fraction", 0.25),
    )
    for iid, why in refusals:
        log_event(db, "INVOICE-REFUSED", f"{iid if iid else 'batch'}: {why}")
    total = sum(actual.values())
    # Route each instance's drift to the bucket that funds its mode.
    net_by_bucket = {}
    for iid, _new_spend, delta in updates:
        bm = bucket_mode(mode_by_id.get(iid), cfg)
        net_by_bucket[bm] = net_by_bucket.get(bm, 0.0) + delta
    if not apply_trueup:
        log_event(
            db, "INVOICE",
            f"observe-only: {len(updates)} instance(s) would move, "
            f"bucket adj ${-net:+.3f}; not applied",
        )
        meta_set(db, "last_invoice_reconcile_at", now)
        db.commit()
        return
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


def _fetch_json(cfg, url):
    """GET and decode. A real User-Agent matters: the hosts sit behind
    Cloudflare, which rejects urllib's default Python-urllib UA with a 403."""
    req = urllib.request.Request(url, headers={"User-Agent": cfg["user_agent"]})
    with urllib.request.urlopen(req, timeout=cfg.get("corpus_timeout_secs", 120)) as resp:
        return json.loads(resp.read())


def sync_corpus(cfg, db):
    """Pull new benchmark reports into the local corpus mirror.

    The estimator used to live behind POST /estimate, which meant one HTTP
    round trip per offer per mode (~400 a tick) against a corpus the API
    truncated to its most recent N reports. Mirroring it locally removes both:
    the whole corpus is available and a tick's estimates cost no network.

    Rows are fetched by id ascending from a watermark, re-reading a small
    overlap because a lower id can commit after a higher one; the upsert makes
    that idempotent. Reports that fail to decode still advance the watermark,
    or they would be re-fetched forever. A failed sync is not fatal: the tick
    proceeds on whatever corpus is already stored.
    """
    overlap = cfg.get("corpus_overlap", 50)
    page = cfg.get("corpus_page_size", 5000)
    watermark = int(meta_get(db, "corpus_watermark", 0) or 0)
    start = max(0, watermark - overlap)
    added = 0
    while True:
        url = (
            f"{cfg['data_base']}/benchmarks"
            f"?id=gt.{start}&order=id.asc&limit={page}"
            f"&select=id,client_version,data"
        )
        try:
            rows = _fetch_json(cfg, url)
        except Exception as e:
            log_event(db, "WARN", f"corpus sync failed at id>{start}: {e!r}")
            break
        if not rows:
            break
        for r in rows:
            sample = estimator.decode_sample(r.get("client_version") or "", r.get("data"))
            if sample is not None:
                db.execute(
                    "INSERT OR REPLACE INTO corpus (id, client_version, gpu, mode, threads, "
                    "cpu_model, gpu_model, scenarios) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (r["id"], sample.client_version, int(sample.gpu), sample.mode,
                     sample.threads, sample.cpu_model, sample.gpu_model,
                     json.dumps(sample.scenarios)),
                )
                added += 1
            start = max(start, r["id"])
        meta_set(db, "corpus_watermark", start)
        db.commit()
        if len(rows) < page:
            break
    total = db.execute("SELECT COUNT(*) FROM corpus").fetchone()[0]
    log_event(db, "CORPUS", f"{total} reports (+{added} this tick, watermark {start})")
    return total


_corpus = None


def load_corpus(db):
    """Decoded corpus for this process. Read once per tick; the controller is
    a fresh process each run, so there is nothing longer-lived to invalidate."""
    global _corpus
    if _corpus is None:
        _corpus = [
            estimator.Sample(
                client_version=r["client_version"],
                gpu=bool(r["gpu"]),
                mode=r["mode"],
                threads=r["threads"],
                cpu_model=r["cpu_model"],
                gpu_model=r["gpu_model"],
                scenarios=json.loads(r["scenarios"]),
            )
            for r in db.execute(
                "SELECT client_version, gpu, mode, threads, cpu_model, gpu_model, scenarios "
                "FROM corpus"
            ).fetchall()
        ]
    return _corpus


def estimate_offer(cfg, db, offer, mode=None):
    """Predict what an offer will achieve. Same response shape the API's
    /estimate returned, so callers are unchanged; the numbers now come from
    the local corpus rather than a round trip."""
    mode_display = estimator.mode_string(mode or cfg["mode"])
    if mode_display is None:
        raise ValueError(f"unknown mode {mode or cfg['mode']!r}")
    corpus = load_corpus(db)
    query = {
        "mode": mode_display,
        "gpu": cfg["gpu"],
        "threads": int(offer.get("cpu_cores_effective") or 0) or None,
        "cpu_model": offer.get("cpu_name"),
        "gpu_model": offer.get("gpu_name"),
    }
    out = estimator.estimate(corpus, query)
    warn_if_divergent(db, cfg, corpus, query, out)
    return out


def warn_if_divergent(db, cfg, corpus, query, out):
    """Flag a thin match that disagrees badly with the wider pool behind it.

    Held-out testing says the narrow stages are usually right and that error
    does not track sample count, so this deliberately does not change what gets
    bought — a rule that overrode them made predictions worse. But a handful of
    cells are genuinely poisoned: RTX 4090 on one EPYC pairing reads 222 G/s
    from three samples, two of them off contended hosts, against 482 G/s from
    the other 75. Those are worth a human look, not an automatic override."""
    factor = cfg.get("divergence_warn_factor", 2.0)
    if not factor or out["prediction_stage"] not in ("exact", "same-gpu-similar-cpu"):
        return
    if out["samples_used"] >= cfg.get("divergence_warn_below_samples", 5):
        return
    narrow = out.get("blended_rate_p50")
    if not narrow:
        return
    # Same GPU, no CPU or thread constraint: the pool the thin match stepped over.
    broad = estimator.estimate(corpus, {**query, "cpu_model": None, "threads": None})
    wide = broad.get("blended_rate_p50")
    if not wide or broad["prediction_stage"] != "same-gpu" or broad["samples_used"] < 5:
        return
    ratio = narrow / wide
    if ratio > factor or ratio < 1.0 / factor:
        log_event(
            db, "DIVERGENT",
            f"{query['gpu_model']} [{query['mode']}]: {out['prediction_stage']} "
            f"n={out['samples_used']} says {narrow/1e9:.0f} G/s but same-gpu "
            f"n={broad['samples_used']} says {wide/1e9:.0f} G/s ({ratio:.2f}x)",
        )


# ---------------------------------------------------------------------------
# Tick phases


def charge_final_interval(db, cfg, row, now, fraction=0.5):
    """Bill the stretch between the last tick and an instance's death.

    Per-tick charging only accrues while an instance is observed `running`,
    so the interval between the last observation and the moment we notice it
    gone was never billed. Instances that lived and died entirely between two
    ticks — 44% of the fleet's history — were therefore recorded at $0.

    We don't know when in that interval it actually stopped, so charge the
    expected value: half, for a uniformly distributed stop time. That is
    unbiased rather than conservative, which is what a budget gate wants.
    Deliberate destroys of an instance seen alive this tick need no call —
    reconcile has already advanced last_charged_at to now.
    """
    dt_h = (now - row["last_charged_at"]) / 3600.0
    if dt_h <= 0:
        return 0.0
    cost = dt_h * row["bid"] * fraction
    db.execute(
        "UPDATE instances SET spend = spend + ?, last_charged_at = ? WHERE vast_id = ?",
        (cost, now, row["vast_id"]),
    )
    charge_bucket(db, bucket_mode(row["mode"], cfg), cost)
    return cost


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
            charge_final_interval(db, cfg, row, now)
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
            charge_final_interval(db, cfg, row, now)
            destroy_instance(cfg, db, vast_id, "preempted", dry)
            log_event(db, "OUTBID", f"instance {vast_id} stopped after {hold_h:.2f}h; reaped")
            continue
        # Charge runtime spend since the last tick (GPU time only accrues
        # while running; a loading instance pays storage, which is noise at
        # our scale and reconciled against invoices).
        if ours_live[vast_id].get("actual_status") == "running":
            dt_h = (now - row["last_charged_at"]) / 3600.0
            cost = dt_h * row["bid"]
            # ever_ran distinguishes "billed $0 because it never ran" from
            # "billed $0 because the invoice hasn't settled" — the invoice
            # true-up needs that to know when a zero charge is trustworthy.
            db.execute(
                "UPDATE instances SET spend = spend + ?, last_charged_at = ?, ever_ran = 1 "
                "WHERE vast_id = ?",
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
            est = estimate_offer(cfg, db, offer_like, mode=m)
        except Exception as e:
            log_event(db, "WARN", f"estimate for pounce {row['vast_id']} failed: {e!r}")
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
            est = estimate_offer(cfg, db, offer_like, mode=m)
        except Exception as e:
            log_event(db, "WARN", f"renew estimate for {row['vast_id']} failed: {e!r}")
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


def corpus_cell_key(gpu_model, cpu_model):
    """The unit of estimator coverage: a GPU model paired with a CPU family.

    Matching keys on the GPU model and, one rung down, on CPU similarity, so
    coverage is really a property of that pair rather than of either alone.
    CPU family (not model) because that is the granularity the corpus can
    actually fill — there are hundreds of CPU models on the market and we buy
    a few hundred instances a month."""
    return (estimator.normalize_gpu_model(gpu_model or "?"), estimator.cpu_match_key(cpu_model))


def corpus_cell_counts(db):
    """Reports per coverage cell, from the local mirror. GPU reports only:
    an explore's value is the hardware pairing it prices, and the CPU-only
    runs in its sweep are priced by the CPU chain regardless of GPU."""
    counts = {}
    for r in db.execute("SELECT gpu_model, cpu_model FROM corpus WHERE gpu = 1").fetchall():
        key = (r["gpu_model"] or "?", estimator.cpu_match_key(r["cpu_model"]))
        counts[key] = counts.get(key, 0) + 1
    return counts


def plan_explore(cfg, db, by_mode, dry):
    """Buy benchmarks for the hardware the estimator cannot price yet.

    Explore used to take the cheapest uncertain offer and cool down on the
    exact (gpu, cpu) string pair. Both parts misfired: cheapest-first kept
    landing on the same few budget cards, and because a popular GPU appears
    with hundreds of different CPUs the pair-level cooldown never bit — 143 of
    850 explores were RTX 3060 and 141 were RTX 3090, a third of the budget
    spent re-measuring two already well-known cards.

    Now that the whole corpus is local the controller can see its own coverage
    directly, so it ranks by how thin a cell is and skips cells already deep
    enough to price confidently. Cheapness only breaks ties."""
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
    offers = {o["id"]: o for mode_offers in by_mode.values() for o, _e, _v in mode_offers}
    counts = corpus_cell_counts(db)
    target = cfg["explore_target_samples"]

    # Cells explored recently, whether or not their reports have landed yet:
    # without this the same thin cell is bought every tick until its benchmarks
    # upload, and a cell whose uploads keep failing would be bought forever.
    cooldown = now - cfg["explore_cooldown_days"] * 86400
    recent = {
        corpus_cell_key(r["gpu_name"], r["cpu_name"])
        for r in db.execute("SELECT gpu_name, cpu_name FROM explored WHERE explored_at > ?",
                            (cooldown,)).fetchall()
    }

    candidates = []
    for offer in offers.values():
        key = corpus_cell_key(offer.get("gpu_name"), offer.get("cpu_name"))
        have = counts.get(key, 0)
        if have >= target or key in recent:
            continue
        # offer id before the dict: it breaks exact (count, bid) ties so the
        # sort never has to compare two offer dicts.
        candidates.append((have, float(offer.get("min_bid", 9e9)), offer["id"], offer, key))
    if not candidates:
        log_event(db, "EXPLORE-SKIP", f"every priceable cell has >= {target} reports")
        return

    # Thinnest cell first; cheapest offer breaks the tie.
    for have, _bid, _id, offer, key in sorted(candidates, key=lambda c: (c[0], c[1], c[2])):
        if slots == 0:
            break
        if key in recent:
            continue   # a cheaper offer for this cell was already taken above
        bid = suggested_bid(offer, cfg, primary_mode(cfg))
        log_event(
            db, "EXPLORE-TARGET",
            f"{key[0]} / {key[1]}: {have} report(s), target {target}",
        )
        create_instance(
            cfg, db, offer, "explore", bid,
            cfg["explore_ttl_minutes"] / 60.0, 0.0, pounce=False, dry=dry, mode=None,
        )
        recent.add(key)   # one buy per cell per tick, not one per matching offer
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

    # 1c. Mirror new benchmark reports before anything estimates. A failed pull
    # is survivable (we keep the stored corpus); an empty one is not, so the
    # tick stops rather than pricing every offer off no data at all.
    if sync_corpus(cfg, db) == 0:
        log_event(db, "ERROR", "benchmark corpus empty; skipping tick")
        return

    # 1d. Renew proven winners before their TTL churns them.
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
                est = estimate_offer(cfg, db, offer, mode=m)
            except Exception as e:
                log_event(db, "WARN", f"estimate [{m}] failed for offer {offer.get('id')}: {e!r}")
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
    """One tick, self-contained: cron calls this directly.

    Locking, log rotation and the run banner used to live in a shell wrapper
    beside this file. None of it was shell-specific, and splitting them meant
    the wrapper had to know where state lived in order to do its half."""
    parser = argparse.ArgumentParser(description="Nice fleet controller")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--live", action="store_true", help="override config dry_run")
    parser.add_argument(
        "--log",
        help="append output to this file instead of stdout, rotating it when "
             "oversized (default: the config's log_path, if it sets one)",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.live:
        cfg["dry_run"] = False

    log_path = args.log or cfg.get("log_path")
    if log_path:
        if not os.path.isabs(log_path):
            log_path = os.path.join(os.path.dirname(os.path.abspath(args.config)), log_path)
        redirect_output_to_log(log_path, cfg.get("log_max_bytes", 5 * 1024 * 1024))

    started = time.strftime("%Y-%m-%d %H:%M:%S %z")
    lock = acquire_tick_lock(cfg["tick_lock_path"] if os.path.isabs(cfg["tick_lock_path"])
                             else os.path.join(os.path.dirname(os.path.abspath(args.config)),
                                               cfg["tick_lock_path"]))
    if lock is None:
        print(f"===== tick {started} SKIPPED: previous tick still running =====")
        return 0

    print(f"===== tick {started} =====")
    rc = 0
    try:
        tick(cfg)
    except Exception:
        traceback.print_exc()
        rc = 1
    print(f"----- exit {rc} at {time.strftime('%Y-%m-%d %H:%M:%S %z')} -----")
    return rc


if __name__ == "__main__":
    sys.exit(main())
