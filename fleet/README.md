# Nice fleet controller

Budgeted explore/exploit over the Vast.ai interruptible market. One tick per
cron invocation: reconcile → accrue budget → probe pounces → search & estimate
offers → benchmark unknown hardware (explore) → buy the best hold-amortized
EV (exploit). Design rationale in `scratchpad/2026-08-vast-fleet/PLAN.md`
(local) and the PR descriptions for fleet plan stages 1–5.

## Setup

```sh
cp config.example.json config.json   # edit: username, api_base if needed
# vastai CLI must be configured: `uvx vastai set api-key <key>`
python3 -m unittest discover .       # 9 tests
python3 controller.py --config config.json   # dry-run tick (default)
```

Cron (10–15 minute cadence is plenty — market hold times are hours):

```cron
*/10 * * * * cd /path/to/fleet && python3 controller.py --config config.json >> tick.log 2>&1
```

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
