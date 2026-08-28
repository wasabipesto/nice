# Nice fleet controller

Budgeted explore/exploit over the Vast.ai interruptible market. One tick per
cron invocation: reconcile → accrue budget → probe pounces → search & estimate
offers → benchmark unknown hardware (explore) → buy the best hold-amortized
EV (exploit).

## Setup

```sh
cp config.example.json config.json   # edit: username, api_base if needed
# vastai CLI must be configured: `uvx vastai set api-key <key>`
uv run -m unittest discover .        # 11 tests
uv run controller.py --config config.json   # dry-run tick (default)
```

Cron (10–15 minute cadence is plenty — market hold times are hours). The
controller handles its own locking, log rotation and run banner, so there is
no wrapper script; cron needs an absolute path to `uv` because its PATH is
minimal, and everything else is resolved relative to the config file:

```cron
*/10 * * * * /path/to/uv run /path/to/fleet/controller.py --config /path/to/state/config.json 2>> /path/to/state/tick.log
```

Set `log_path` in the config to have the controller write the tick log itself.
The `2>>` catches the narrow window before Python starts — a missing `uv`, an
unresolvable dependency — which the controller cannot log for itself.

## Trust ramp

1. **Week 1 — dry run** (`"dry_run": true`, the default): every tick plans
   and prints creates/destroys without performing any. Read `tick.log`,
   sanity-check the EV rankings and explore picks.
2. **Explore-only**: set `"max_exploit_instances": 0`, `--live`. Spends
   pennies benchmarking unknown hardware; seeds the estimator.
3. **Small exploit**: `"max_exploit_instances": 1` with the default budget.
   Watch realized vs predicted for a week (`SUMMARY` lines + the server's
   benchmarks/telemetry tables).
4. Raise caps only after deliberately testing the failure drills below.

## Failure drills (run before trusting it with real budget)

- Kill the controller mid-tick; next tick must reconcile cleanly.
- Create an unlabeled/foreign-labeled instance manually; the controller must
  leave it alone. Create one with the fleet label; it must be destroyed as
  an orphan.
- Touch the kill-switch file (`KILL` next to the config by default): the
  next tick destroys all fleet instances and buys nothing until removed.
- Let a TTL lapse with the controller stopped; on restart the instance is
  destroyed on the first reconcile.

## Budget model (token bucket)

Budget accrues continuously at `accrual_usd_per_month` (default $30) into a
bucket capped at `bucket_cap_usd` (default $7). Ordinary buys require the
bucket above `reserve_fraction` of cap. Exceptional deals ("pounces") may
spend below the reserve when they beat the trailing 3-day median EV by
`pounce_multiplier` (default 1.4×): they start on a `pounce_probe_hours`
TTL and are extended only once the estimator — refreshed by the instance's
own uploaded benchmark — confirms the buy at a trustworthy prediction stage.
Worst-case month ≈ accrual + one bucket. Runtime spend is charged to the
bucket every tick from bid × elapsed; reconcile against Vast invoices
periodically (`uvx vastai show invoices`) until that's automated.

### Manual bucket adjustments (kickstart / correction)

There is no CLI for this by design; adjust the ledger row directly and always
tag it so the `events` log stays a complete audit trail. Use a **relative**
delta (`balance + N`), never an absolute set — the per-tick accrue writes an
absolute balance, so apply the credit **between ticks** (mid-slot, not near
`:00`/`:10`) to avoid a race clobbering it. Tag `kind = 'MANUAL-CREDIT'`:

```sh
sqlite3 fleet.sqlite3 "
UPDATE bucket SET balance = balance + 2.0 WHERE id = 1;
INSERT INTO events (ts, kind, detail)
VALUES (strftime('%s','now'), 'MANUAL-CREDIT', '+\$2.00 kickstart: <reason>');"
```

A credit above `$0` re-enables explores; keep it below the reserve line
(`reserve_fraction × bucket_cap_usd`) if you want to avoid also unblocking a
wave of ordinary exploit buys. `MANUAL-CREDIT` is a one-time injection outside
the accrual bound, so note why.

## Notes / known gaps

- **First live explore run validates the launch incantation.** The GPU
  image's ENTRYPOINT is `nice_client`, so instances override to bash and
  run `onstart_*` templates from the config. If Vast's create semantics
  differ from expectation, fix the config strings, not the code.
- The heat guard and pounce baseline need history (≈20 samples) before they
  act; the first days run permissive-but-reserve-gated.
- Realized-throughput confirmation currently rides on the benchmark-upload →
  `/estimate` loop; per-field telemetry correlation is a future refinement.
- Explore instances are cheap but not free: `explore_per_day` × ~2–5 min of
  the cheapest matching offer (well under $0.05/day at defaults).
