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
    "mode": "niceonly",
    "gpu": True,
    "max_exploit_instances": 2,
    "exploit_stages": ["exact", "same-gpu-similar-cpu", "same-gpu", "same-cpu-scaled"],
    "exploit_min_confidence": 40,
    "exploit_ttl_hours": 6.0,
    "image": "ghcr.io/wasabipesto/nice_client:latest-gpu",
    "disk_gb": 30,
    # Instances launch in ssh runtype: Vast runs its own init and executes
    # onstart_cmd in a shell, so the image ENTRYPOINT is bypassed and the
    # templates below can call nice_client (on PATH in the image) directly.
    "onstart_exploit": (
        "nice_client {mode} --gpu --benchmark --benchmark-upload --no-progress "
        "--api-base {api_base} --username {username} --threads {threads} && "
        "exec nice_client {mode} --gpu --repeat --telemetry --no-progress "
        "--api-base {api_base} --username {username} --threads {threads}"
    ),
    # After the benchmark uploads, the explore instance retires itself using
    # the instance-scoped API key Vast injects into the environment; the
    # controller's TTL remains the backstop if that ever fails.
    "onstart_explore": (
        "nice_client {mode} --gpu --benchmark --benchmark-upload --no-progress "
        "--api-base {api_base} --username {username} --threads {threads} ; "
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
CREATE TABLE IF NOT EXISTS bucket (
    id INTEGER PRIMARY KEY CHECK (id = 1),
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
    spend REAL NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS explored (
    gpu_name TEXT NOT NULL,
    cpu_name TEXT NOT NULL,
    explored_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS ev_seen (
    ts REAL NOT NULL,
    ev REAL NOT NULL
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
# Pure budget / economics logic (unit-tested)


def accrue_bucket(balance, updated_at, now, cfg):
    """Token bucket accrual: continuous drip at the sustained rate, capped."""
    hours = max(0.0, (now - updated_at) / 3600.0)
    rate = cfg["accrual_usd_per_month"] / (30.0 * 24.0)
    return min(balance + hours * rate, cfg["bucket_cap_usd"])


def effective_rate(p25_rate, e_hold_hours, setup_hours, lost_hours):
    """Hold-amortized throughput: the spin-up tax. Setup and the expected
    lost interval at preemption are amortized over the type's expected
    hold, so churn-prone types pay automatically."""
    if p25_rate is None or p25_rate <= 0 or e_hold_hours <= 0:
        return 0.0
    return p25_rate * e_hold_hours / (e_hold_hours + setup_hours + lost_hours)


def offer_ev(offer, estimate, e_hold_hours, cfg):
    """Numbers per dollar, using the conservative P25 of the prediction."""
    bid = suggested_bid(offer, cfg)
    if bid <= 0:
        return 0.0
    rate = effective_rate(
        estimate.get("blended_rate_p25"),
        e_hold_hours,
        cfg["setup_hours"],
        cfg["lost_interval_hours"],
    )
    return rate / bid


def suggested_bid(offer, cfg):
    """Ride the floor: a small multiplier on the offer's minimum bid.
    Premium buys hours, not days (2026-07 market study), so protection
    comes from type selection and buffer depth, not big overbids."""
    return round(float(offer.get("min_bid", offer.get("dph_total", 0))) * cfg["bid_multiplier"], 5)


def instance_alive(live_row, cfg, age_minutes=0.0):
    """Is this live-listed instance still doing useful work? Running always
    counts; transitional states count only within the stuck grace period."""
    status = live_row.get("actual_status") or "created"
    if status == "running":
        return True
    return status in cfg["alive_statuses"] and age_minutes < cfg["stuck_grace_minutes"]


def exploit_allowed(est, cfg):
    """Exploit buys only trust estimates from the allowlisted prediction
    stages at sufficient confidence. Floor/none-stage hardware — where a
    single wild sample can produce an absurd EV with no spread to hedge it
    — is the explore slice's job, not a purchase signal."""
    return (
        est.get("prediction_stage") in cfg["exploit_stages"]
        and est.get("confidence", 0) >= cfg["exploit_min_confidence"]
    )


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


def trailing_median_ev(db, days=3.0):
    rows = db.execute(
        "SELECT ev FROM ev_seen WHERE ts > ?", (time.time() - days * 86400,)
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


def search_offers(cfg):
    return vast_client(cfg).search_offers(
        query=cfg["offer_query"], type=cfg["offer_type"], limit=200
    )


def show_instances(cfg):
    return vast_client(cfg).show_instances()


def create_instance(cfg, db, offer, purpose, bid, ttl_hours, ev, pounce, dry):
    onstart_tpl = cfg["onstart_exploit"] if purpose == "exploit" else cfg["onstart_explore"]
    onstart = onstart_tpl.format(
        mode=cfg["mode"],
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
        return
    if not isinstance(result, dict) or not result.get("success"):
        log_event(db, "ERROR", f"create rejected for offer {offer['id']}: {result!r}")
        return
    vast_id = result.get("new_contract")
    now = time.time()
    db.execute(
        "INSERT INTO instances (vast_id, label, purpose, gpu_name, cpu_name, bid, "
        "ev_predicted, pounce, created_at, ttl_at, last_charged_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            vast_id, label, purpose, offer.get("gpu_name"), offer.get("cpu_name"),
            bid, ev, int(pounce), now, now + ttl_hours * 3600, now,
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


def api_estimate(cfg, offer):
    body = {
        "mode": cfg["mode"],
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
            db.execute("UPDATE bucket SET balance = balance - ? WHERE id = 1", (cost,))
        else:
            db.execute(
                "UPDATE instances SET last_charged_at = ? WHERE vast_id = ?", (now, vast_id)
            )
        if now >= row["ttl_at"]:
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
        offer_like = {
            "cpu_name": row["cpu_name"],
            "gpu_name": row["gpu_name"],
            "min_bid": row["bid"] / cfg["bid_multiplier"],
        }
        try:
            est = api_estimate(cfg, offer_like)
        except OSError as e:
            log_event(db, "WARN", f"estimate for pounce {row['vast_id']} failed: {e}")
            continue
        ev = offer_ev(offer_like, est, e_hold_hours(db, cfg, row["gpu_name"]), cfg)
        trailing = trailing_median_ev(db)
        good_stage = est.get("prediction_stage") in ("exact", "same-gpu-similar-cpu")
        if good_stage and pounce_eligible(ev, trailing, cfg):
            db.execute(
                "UPDATE instances SET confirmed = 1, ttl_at = ? WHERE vast_id = ?",
                (now + cfg["exploit_ttl_hours"] * 3600, row["vast_id"]),
            )
            log_event(
                db, "POUNCE-CONFIRMED",
                f"instance {row['vast_id']} stage={est.get('prediction_stage')} ev={ev:.3e}",
            )
        elif now >= row["ttl_at"]:
            destroy_instance(cfg, db, row["vast_id"], "pounce-unconfirmed", dry)
    db.commit()


def record_market(db, offers_by_est):
    """Persist per-type p10 floors and the EV distribution for the heat
    guard and the pounce baseline."""
    now = time.time()
    by_type = {}
    for offer, _est, ev in offers_by_est:
        by_type.setdefault(offer.get("gpu_name") or "?", []).append(
            float(offer.get("min_bid", offer.get("dph_total", 0)))
        )
        if ev > 0:
            db.execute("INSERT INTO ev_seen (ts, ev) VALUES (?, ?)", (now, ev))
    for gpu_name, prices in by_type.items():
        prices.sort()
        p10 = prices[max(0, int(len(prices) * 0.10) - 1)] if len(prices) > 1 else prices[0]
        db.execute(
            "INSERT INTO type_floor (ts, gpu_name, p10_dph) VALUES (?, ?, ?)",
            (now, gpu_name, p10),
        )
    # Keep the market tables bounded.
    cutoff = now - 30 * 86400
    db.execute("DELETE FROM ev_seen WHERE ts < ?", (cutoff,))
    db.execute("DELETE FROM type_floor WHERE ts < ?", (cutoff,))
    db.commit()


def plan_explore(cfg, db, offers_by_est, dry):
    now = time.time()
    balance = db.execute("SELECT balance FROM bucket WHERE id = 1").fetchone()["balance"]
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
    cooldown = now - cfg["explore_cooldown_days"] * 86400
    for offer, est, ev in sorted(offers_by_est, key=lambda t: float(t[0].get("min_bid", 9e9))):
        if slots == 0:
            break
        uncertain = (
            est.get("confidence", 0) < cfg["explore_confidence_threshold"]
            or est.get("prediction_stage") in cfg["explore_stages"]
        )
        if not uncertain:
            continue
        seen = db.execute(
            "SELECT COUNT(*) FROM explored WHERE gpu_name = ? AND cpu_name = ? AND explored_at > ?",
            (offer.get("gpu_name") or "?", offer.get("cpu_name") or "?", cooldown),
        ).fetchone()[0]
        if seen:
            continue
        bid = suggested_bid(offer, cfg)
        create_instance(
            cfg, db, offer, "explore", bid,
            cfg["explore_ttl_minutes"] / 60.0, ev, pounce=False, dry=dry,
        )
        slots -= 1
    db.commit()


def plan_exploit(cfg, db, offers_by_est, dry):
    now = time.time()
    balance = db.execute("SELECT balance FROM bucket WHERE id = 1").fetchone()["balance"]
    reserve = cfg["bucket_cap_usd"] * cfg["reserve_fraction"]
    active = db.execute(
        "SELECT COUNT(*) FROM instances WHERE destroyed_at IS NULL AND purpose = 'exploit'"
    ).fetchone()[0]
    trailing = trailing_median_ev(db)

    ranked = sorted(
        (t for t in offers_by_est if t[2] > 0 and exploit_allowed(t[1], cfg)),
        key=lambda t: t[2],
        reverse=True,
    )
    skipped = sum(1 for t in offers_by_est if t[2] > 0 and not exploit_allowed(t[1], cfg))
    if skipped:
        log_event(db, "STAGE-SKIP", f"{skipped} offers below exploit trust bar (explore's job)")
    for offer, est, ev in ranked:
        if active >= cfg["max_exploit_instances"]:
            break
        gpu_name = offer.get("gpu_name") or "?"
        prices = sorted(
            float(o.get("min_bid", o.get("dph_total", 9e9)))
            for o, _e, _v in offers_by_est
            if (o.get("gpu_name") or "?") == gpu_name
        )
        current_p10 = prices[max(0, int(len(prices) * 0.10) - 1)] if prices else 9e9
        if not heat_ok(db, cfg, gpu_name, current_p10):
            log_event(db, "HEAT-SKIP", f"{gpu_name} running hot; skipping this tick")
            continue
        bid = suggested_bid(offer, cfg)
        ordinary = balance > reserve
        pounce = not ordinary and pounce_eligible(ev, trailing, cfg) and balance > 0
        if pounce:
            probe_cost = bid * cfg["pounce_probe_hours"]
            if probe_cost > balance / 2:
                continue
        if not ordinary and not pounce:
            break  # bucket at/below reserve and nothing exceptional: hold
        ttl = cfg["pounce_probe_hours"] if pounce else cfg["exploit_ttl_hours"]
        create_instance(cfg, db, offer, "exploit", bid, ttl, ev, pounce, dry)
        active += 1
        balance -= bid * ttl  # planning estimate only; real charge accrues per tick
    db.commit()


def tick(cfg):
    dry = cfg["dry_run"]
    db = open_db(cfg["db_path"])
    now = time.time()

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

    # 2. Accrue the bucket.
    row = db.execute("SELECT balance, updated_at FROM bucket WHERE id = 1").fetchone()
    balance = accrue_bucket(row["balance"], row["updated_at"], now, cfg)
    db.execute("UPDATE bucket SET balance = ?, updated_at = ? WHERE id = 1", (balance, now))
    log_event(db, "BUCKET", f"balance ${balance:.3f} / cap ${cfg['bucket_cap_usd']:.2f}")

    # 3. Pounce probes come before new purchases.
    extend_or_probe_pounces(cfg, db, dry)

    # 4. Market snapshot + estimates.
    try:
        offers = search_offers(cfg)
    except Exception as e:
        log_event(db, "ERROR", f"offer search failed: {e!r}")
        return
    offers_by_est = []
    for offer in offers:
        try:
            est = api_estimate(cfg, offer)
        except OSError as e:
            log_event(db, "WARN", f"estimate failed for offer {offer.get('id')}: {e}")
            continue
        ev = offer_ev(offer, est, e_hold_hours(db, cfg, offer.get("gpu_name") or "?"), cfg)
        offers_by_est.append((offer, est, ev))
    record_market(db, offers_by_est)
    log_event(db, "MARKET", f"{len(offers_by_est)} offers estimated")

    # 5. Explore, then exploit.
    plan_explore(cfg, db, offers_by_est, dry)
    plan_exploit(cfg, db, offers_by_est, dry)

    # Tick summary.
    active = db.execute(
        "SELECT COUNT(*), COALESCE(SUM(spend), 0) FROM instances WHERE destroyed_at IS NULL"
    ).fetchone()
    month_spend = db.execute(
        "SELECT COALESCE(SUM(spend), 0) FROM instances WHERE created_at > ?",
        (now - 30 * 86400,),
    ).fetchone()[0]
    log_event(
        db, "SUMMARY",
        f"active={active[0]} active_spend=${active[1]:.3f} 30d_spend=${month_spend:.3f} "
        f"bucket=${balance:.3f}{' [DRY RUN]' if dry else ''}",
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
