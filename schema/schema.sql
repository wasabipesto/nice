-- DROP TABLES IN REVERSE ORDER
DROP TABLE IF EXISTS submissions;
DROP TABLE IF EXISTS claims;
DROP TABLE IF EXISTS fields;
DROP TABLE IF EXISTS chunks;
DROP TABLE IF EXISTS bases;

-- BASES: ENTRIE BASE RANGE
CREATE TABLE bases (
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
CREATE TABLE chunks (
    id SERIAL PRIMARY KEY,
    base_id INTEGER NOT NULL REFERENCES bases(id),
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
CREATE TABLE fields (
    id BIGSERIAL PRIMARY KEY,
    base_id INTEGER NOT NULL REFERENCES bases(id),
    chunk_id INTEGER REFERENCES chunks(id),
    range_start DECIMAL NOT NULL,
    range_end DECIMAL NOT NULL,
    range_size DECIMAL NOT NULL,
    last_claim_time TIMESTAMPTZ,
    canon_submission_id INTEGER,
    check_level INTEGER NOT NULL DEFAULT 0,
    prioritize BOOLEAN NOT NULL DEFAULT 'false'
);

-- CLAIMS: LOG OF CLAIM REQUESTS
CREATE TABLE claims (
    id BIGSERIAL PRIMARY KEY,
    field_id INTEGER NOT NULL REFERENCES fields(id),
    search_mode VARCHAR NOT NULL,
    claim_time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    user_ip VARCHAR NOT NULL
);

-- SUBMISSIONS: LOG OF ALL VALIDATED SUIBMISSIONS
CREATE TABLE submissions (
    id BIGSERIAL PRIMARY KEY,
    claim_id INTEGER NOT NULL REFERENCES claims(id),
    field_id INTEGER NOT NULL REFERENCES fields(id),
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

-- POSTGREST USER ACCESS
CREATE ROLE web_anon NOLOGIN;
GRANT SELECT ON bases TO web_anon;
GRANT SELECT ON chunks TO web_anon;
GRANT SELECT ON fields TO web_anon;
GRANT SELECT ON claims TO web_anon;
GRANT SELECT ON submissions TO web_anon;

-- ADDITIONAL INDEXES
CREATE INDEX IF NOT EXISTS idx_fields_base_id ON fields(base_id);
CREATE INDEX IF NOT EXISTS idx_fields_range_start ON fields(range_start);
CREATE INDEX IF NOT EXISTS idx_fields_range_end ON fields(range_end);
CREATE INDEX IF NOT EXISTS idx_fields_check_level ON fields(check_level);
CREATE INDEX IF NOT EXISTS idx_fields_last_claim_time ON fields(last_claim_time);
CREATE INDEX IF NOT EXISTS idx_fields_base_check_level ON fields(base_id, check_level);
CREATE INDEX IF NOT EXISTS idx_fields_range_start_end ON fields(range_start, range_end);
CREATE INDEX IF NOT EXISTS idx_fields_base_range ON fields(base_id, range_start, range_end);
CREATE INDEX IF NOT EXISTS idx_fields_check_level_range_size_last_claim_time_id ON fields(check_level, range_size, last_claim_time, id);
CREATE INDEX IF NOT EXISTS idx_fields_canon_submission ON fields(canon_submission_id) WHERE canon_submission_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_submissions_search_mode_field_id ON submissions(search_mode, field_id);
CREATE INDEX IF NOT EXISTS idx_submissions_field_mode_disq ON submissions(field_id, search_mode, disqualified);
CREATE INDEX IF NOT EXISTS idx_submissions_id_field ON submissions(id, field_id);
