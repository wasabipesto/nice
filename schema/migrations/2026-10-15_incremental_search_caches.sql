-- Incremental search caches.
-- Apply manually; schema.sql has been updated to match.
--
-- The scheduled jobs used to rebuild cache_search_rate_daily and
-- cache_search_leaderboard every run with two joins over all of submissions
-- and fields, each a full read of both tables. They now keep all-time daily
-- totals per (date, mode, user) in cache_search_daily_totals, add only the
-- submissions that arrived since the last run, and derive both website tables
-- from the totals. The website tables and what they contain are unchanged.
--
-- job_state.cache_last_submission_id is the highest submission id counted
-- into the totals. Unlike last_processed_submission_id it is exact: the totals
-- and the cursor always commit together, so nothing is counted twice. NULL
-- means the totals have never been built.
--
-- The backfill below is one full read of submissions and fields, the same
-- work as `just jobs-full` does for the totals. Apply it between scheduled
-- runs, not near :50. It runs in one transaction, so if anything fails nothing
-- changes. Apply it before deploying the jobs binary that reads the new
-- column; that binary refuses to start without it.

BEGIN;

CREATE TABLE cache_search_daily_totals (
    date DATE NOT NULL,
    search_mode VARCHAR NOT NULL,
    username VARCHAR NOT NULL,
    total_range DECIMAL NOT NULL,
    PRIMARY KEY (date, search_mode, username)
);

ALTER TABLE job_state ADD COLUMN cache_last_submission_id BIGINT;

-- Day boundaries in UTC, whatever the server's default time zone: the jobs
-- binary's connections run in UTC (diesel sets it on connect), so that is how
-- the website tables have always been bucketed, and how the totals the jobs
-- add from now on will be.
SET LOCAL TIME ZONE 'UTC';

DO $$
DECLARE
    up_to BIGINT;
BEGIN
    -- Hold back the newest 10,000 ids, the jobs binary's
    -- WATERMARK_SAFETY_MARGIN: they may belong to submissions that have not
    -- committed yet. The next scheduled run counts them.
    SELECT GREATEST(COALESCE(MAX(id), 0) - 10000, 0) INTO up_to FROM submissions;

    INSERT INTO cache_search_daily_totals (date, search_mode, username, total_range)
    SELECT DATE(s.submit_time), s.search_mode, s.username, SUM(f.range_size)
    FROM submissions s
    JOIN fields f ON f.id = s.field_id
    WHERE s.id <= up_to AND NOT s.disqualified
    GROUP BY 1, 2, 3;

    UPDATE job_state SET cache_last_submission_id = up_to WHERE id = 1;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'job_state has no row with id = 1; apply 2026-08-20_job_state.sql first';
    END IF;
END
$$;

COMMIT;
