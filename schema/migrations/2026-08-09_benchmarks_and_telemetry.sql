-- Benchmark uploads and submission telemetry.
-- Apply manually; schema.sql has been updated to match.

-- BENCHMARKS: uploaded --benchmark sweep reports.
-- The full versioned report (scenarios, hardware, environment, API latency,
-- score) lives in `data` as produced by the client; `client_version` is
-- extracted for cheap filtering. Deliberately NOT granted to web_anon:
-- rows carry user_ip.
CREATE TABLE benchmarks (
    id BIGSERIAL PRIMARY KEY,
    submit_time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    username VARCHAR NOT NULL,
    user_ip VARCHAR NOT NULL,
    client_version VARCHAR NOT NULL,
    data JSONB NOT NULL
);
CREATE INDEX idx_benchmarks_submit_time ON benchmarks(submit_time);
CREATE INDEX idx_benchmarks_client_version ON benchmarks(client_version);

-- TELEMETRY: optional per-submission hardware/config context sent by clients
-- running with --telemetry. NULL for clients that don't opt in.
ALTER TABLE submissions ADD COLUMN telemetry JSONB;
