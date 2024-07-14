-- DROP TABLES IN REVERSE ORDER
DROP TABLE IF EXISTS submission;
DROP TABLE IF EXISTS claim;
DROP TABLE IF EXISTS field;
DROP TABLE IF EXISTS chunk;
DROP TABLE IF EXISTS base;
-- BASES: ENTRIE BASE RANGE
CREATE TABLE base (
    -- ID is the acual base
    id INTEGER PRIMARY KEY,
    range_start DECIMAL NOT NULL,
    range_end DECIMAL NOT NULL,
    range_size DECIMAL NOT NULL,
    checked_detailed DECIMAL NOT NULL DEFAULT 0,
    checked_niceonly DECIMAL NOT NULL DEFAULT 0,
    minimum_cl INTEGER NOT NULL DEFAULT 0,
    niceness_mean REAL,
    niceness_stdev REAL,
    distribution JSONB NOT NULL DEFAULT '[]',
    numbers JSONB NOT NULL DEFAULT '[]'
);
-- CHUNKS: AGGREGATE FIELDS FOR ANALYTICS
CREATE TABLE chunk (
    id SERIAL PRIMARY KEY,
    base_id INTEGER NOT NULL REFERENCES base(id),
    range_start DECIMAL NOT NULL,
    range_end DECIMAL NOT NULL,
    range_size DECIMAL NOT NULL,
    checked_detailed DECIMAL NOT NULL DEFAULT 0,
    checked_niceonly DECIMAL NOT NULL DEFAULT 0,
    minimum_cl INTEGER NOT NULL DEFAULT 0,
    niceness_mean REAL,
    niceness_stdev REAL,
    distribution JSONB NOT NULL DEFAULT '[]',
    numbers JSONB NOT NULL DEFAULT '[]'
);
-- FIELDS: INDIVIDUAL SEARCH RANGES
CREATE TABLE field (
    id BIGSERIAL PRIMARY KEY,
    base_id INTEGER NOT NULL REFERENCES base(id),
    chunk_id INTEGER REFERENCES chunk(id),
    range_start DECIMAL NOT NULL,
    range_end DECIMAL NOT NULL,
    range_size DECIMAL NOT NULL,
    last_claim_time TIMESTAMPTZ,
    canon_submission_id INTEGER,
    check_level INTEGER NOT NULL DEFAULT 0,
    prioritize BOOLEAN NOT NULL DEFAULT 'false'
);
-- CLAIMS: LOG OF CLAIM REQUESTS
CREATE TABLE claim (
    id BIGSERIAL PRIMARY KEY,
    field_id INTEGER NOT NULL REFERENCES field(id),
    search_mode VARCHAR NOT NULL,
    claim_time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    user_ip VARCHAR NOT NULL
);
-- SUBMISSIONS: LOG OF ALL VALIDATED SUIBMISSIONS
CREATE TABLE submission (
    id BIGSERIAL PRIMARY KEY,
    claim_id INTEGER NOT NULL REFERENCES claim(id),
    field_id INTEGER NOT NULL REFERENCES field(id),
    search_mode VARCHAR NOT NULL,
    submit_time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    elapsed_secs REAL NOT NULL,
    username VARCHAR NOT NULL,
    user_ip VARCHAR NOT NULL,
    client_version VARCHAR NOT NULL,
    disqualified BOOLEAN NOT NULL DEFAULT 'false',
    distribution JSONB,
    numbers JSONB NOT NULL DEFAULT '[]'
);