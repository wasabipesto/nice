-- BASES: ENTRIE BASE RANGE
DROP TABLE IF EXISTS base;
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
DROP TABLE IF EXISTS chunk;
CREATE TABLE chunk (
    id SERIAL PRIMARY KEY,
    base_id INTEGER NOT NULL,
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
DROP TABLE IF EXISTS field;
CREATE TABLE field (
    id BIGSERIAL PRIMARY KEY,
    base_id INTEGER NOT NULL,
    chunk_id INTEGER,
    range_start DECIMAL NOT NULL,
    range_end DECIMAL NOT NULL,
    range_size DECIMAL NOT NULL,
    last_claim_time TIMESTAMPTZ,
    canon_submission_id INTEGER,
    check_level INTEGER NOT NULL DEFAULT 0,
    prioritize BOOLEAN NOT NULL DEFAULT 'false'
);
-- CLAIMS: LOG OF CLAIM REQUESTS
DROP TABLE IF EXISTS claim;
CREATE TABLE claim (
    id BIGSERIAL PRIMARY KEY,
    field_id INTEGER NOT NULL,
    search_mode VARCHAR,
    claim_time TIMESTAMPTZ,
    user_ip VARCHAR,
    user_agent VARCHAR
);
-- SUBMISSIONS: LOG OF ALL VALIDATED SUIBMISSIONS
DROP TABLE IF EXISTS submission;
CREATE TABLE submission (
    id BIGSERIAL PRIMARY KEY,
    claim_id INTEGER NOT NULL,
    field_id INTEGER NOT NULL,
    search_mode VARCHAR NOT NULL,
    submit_time TIMESTAMPTZ,
    elapsed_secs INTEGER,
    username VARCHAR NOT NULL,
    user_ip VARCHAR,
    user_agent VARCHAR,
    client_version VARCHAR,
    disqualified BOOLEAN NOT NULL DEFAULT 'false',
    distribution JSONB,
    numbers JSONB NOT NULL DEFAULT '[]'
);