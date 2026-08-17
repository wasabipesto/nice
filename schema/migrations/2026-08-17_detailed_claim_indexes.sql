-- Indexes for the detailed claim path (issue #98).
-- Apply manually; schema.sql has been updated to match.
--
-- Both are built CONCURRENTLY: they cover the live claim path on a large table, and a
-- plain CREATE INDEX would hold a write lock against every incoming claim while it
-- builds. CONCURRENTLY cannot run inside a transaction block, so run this file with
-- autocommit (psql does this by default; do not wrap it in BEGIN/COMMIT).
--
-- Neither index is required for correctness — the queries return the same rows without
-- them, just far more slowly. If the build fails partway it leaves an INVALID index
-- behind; DROP INDEX and retry.

-- Detailed claims select `check_level <= 1 ORDER BY id`. The existing composite index
-- idx_fields_check_level_range_size_last_claim_time_id cannot produce that ordering
-- across two check_level values, so the planner falls back to walking fields_pkey and
-- filtering out every completed row ahead of the frontier. This is the `<= 1` analogue
-- of idx_fields_cl0_id, which is what makes the nice-only claim fast.
--
-- Note this index covers every check_level 0 row as well, so it substantially overlaps
-- idx_fields_cl0_id. Keeping both costs disk; dropping idx_fields_cl0_id would make
-- nice-only claims use this (larger) index instead. Left to a follow-up with production
-- size numbers in hand.
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_fields_cl1_id
    ON fields(id) WHERE check_level <= 1;

-- Frontier chunk selection now asks "does this chunk still hold a claimable field?" for
-- each candidate chunk in id order. Without this index that probe reads every field of
-- every exhausted chunk it passes over, and the number of exhausted chunks grows between
-- `jobs` runs (a chunk whose work is finished still looks under-explored until
-- checked_detailed is recomputed).
--
-- Deliberately covers all four predicate columns and is deliberately NOT partial. The
-- point is to make the probe an index-only scan with every predicate as an index
-- condition: measured 13.9ms -> 0.36ms against a 1.2M row fixture. A partial
-- `WHERE check_level <= 1` variant is smaller but the planner consistently preferred the
-- existing idx_fields_chunk_id over it, leaving the probe at ~20ms; the covering index
-- wins on cost honestly rather than by hint.
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_fields_chunk_claim_cover
    ON fields(chunk_id, last_claim_time, range_size, check_level);
