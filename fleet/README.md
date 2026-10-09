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

## Client version

The fleet prices offers with one client version's benchmarks and launches that
same version, so the two cannot drift apart:

- `client_version: "auto"` (the default) follows the newest `X.Y.Z` release
  image in the registry on the controller's major line (`CLIENT_MAJOR`, 3). A
  version's tag appears there only once its image is pushed, so the fleet
  never prices a version it cannot launch yet. The lookup runs at most hourly;
  a failed one keeps the last version found, and with none ever found the
  tick buys nothing.
- A version string (`"3.4.5"`) pins it, e.g. to hold the fleet on a release.
- `image` names the repository only. Instances run `<image>:<version>-gpu`
  (`<image>:<version>` for a CPU fleet), never a moving tag like `3-gpu` or
  `latest-gpu`; a tag written in `image` is ignored with a warning.
- Each instance records the version it was launched with, and renewal and
  pounce confirmation price it at that version. Instances launched before
  versions were recorded are priced at the current one.
- Explore counts coverage from the current version's reports only, so after a
  release it re-measures hardware that only older versions have measured.
- A `VERSION` event logs each change, and `SUMMARY` lines name the image.

This relies on a release's tag matching the workspace version the client
reports (the `v3.4.5` tag carries `version = "3.4.5"`).

## Host CPU client

`host_cpu_client: true` (off by default; per mode inside `exploit_modes`)
makes each exploit also run a CPU client of its own mode on the host's spare
threads: a nice-only deployment runs nice-only on the CPU, a detailed one
detailed. It starts after the benchmark sweep and exits with the container
when the GPU client does. It gets the host's threads less
`host_cpu_reserve_threads` (2), and none on a host where that leaves fewer than
`host_cpu_min_threads` (4). The estimator does not price this work yet, so it is
a free addition to the GPU's, and its submissions carry the instance's
`CONTAINER_ID` like the GPU client's.

## Notes / known gaps

- **First live explore run validates the launch incantation.** The GPU
  image's ENTRYPOINT is `nice_client`, so instances override to bash and
  run `onstart_*` templates from the config. If Vast's create semantics
  differ from expectation, fix the config strings, not the code.
- The heat guard and pounce baseline need history (≈20 samples) before they
  act; the first days run permissive-but-reserve-gated.
- Realized-throughput confirmation currently rides on the benchmark-upload →
  `/estimate` loop; per-field telemetry correlation is a future refinement.
- Explores delete themselves after their benchmark sweep, usually between
  ticks, so reconcile never sees them running. It checks a vanished explore's
  benchmark uploads: one with uploads is recorded `retired` and billed at its
  bid up to its last upload, and its `RETIRED` event names any benchmark that
  didn't upload. One with none, or a failed lookup, is recorded `preempted`.
- Explore instances are cheap but not free: `explore_per_day` × ~2–5 min of
  the cheapest matching offer (well under $0.05/day at defaults).
