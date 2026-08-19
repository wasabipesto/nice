-- Watermark for incremental scheduled jobs.
-- Apply manually; schema.sql has been updated to match.
--
-- The jobs binary processes only submissions with id above this watermark
-- (consensus for new detailed submissions, chunk/base stats for chunks with
-- any new submission), then advances it. `just jobs-full` ignores it and
-- sweeps everything - required after manual changes that create no new
-- submission, such as disqualifying one.

CREATE TABLE job_state (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    last_processed_submission_id BIGINT NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Initialize to the current maximum submission id: stats are assumed current
-- as of applying this migration, since jobs has been running on schedule.
-- If in doubt, run `just jobs-full` once after deploying.
INSERT INTO job_state (id, last_processed_submission_id)
SELECT 1, COALESCE(MAX(id), 0) FROM submissions;
