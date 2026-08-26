-- Cache the points plotted by the "Notably Nice Numbers" chart on the website.
--
-- The chart used to be drawn from `bases.numbers` directly: every base's full
-- top-10k list, 686 KB gzipped, ~81k points, over 99% of which land on a pixel
-- another point already covers. This table holds only the points that are
-- visually distinguishable, which is a few hundred rows.
--
-- Thinning is exact in y and in colour, because a point's niceness is exactly
-- num_uniques / base and its colour is base - num_uniques: both are fixed by
-- (base, num_uniques). So only x needs quantizing, at 70 buckets per decade
-- against the ~61 pixels per decade the chart has. Points within two uniques of
-- a nice number skip bucketing and are all kept; that cannot collide with the
-- bucketed key space because off_by is constant within a (base, num_uniques)
-- group.
--
-- Repopulated by the scheduled jobs on every run. Safe to apply while the API
-- and jobs are running: nothing else reads or writes this table.

CREATE TABLE IF NOT EXISTS cache_notable_numbers (
    base        INTEGER NOT NULL,
    number      DECIMAL NOT NULL,
    num_uniques INTEGER NOT NULL,
    off_by      INTEGER NOT NULL,
    niceness    REAL    NOT NULL,
    PRIMARY KEY (base, number)
);

GRANT SELECT ON cache_notable_numbers TO web_anon;

-- Populate immediately, so the chart works before the next scheduled run.
INSERT INTO cache_notable_numbers (base, number, num_uniques, off_by, niceness)
SELECT DISTINCT ON (b.id, n.num_uniques, n.bucket)
    b.id, n.number, n.num_uniques, b.id - n.num_uniques, n.niceness
FROM bases b
CROSS JOIN LATERAL (
    SELECT
        (e->>'number')::decimal AS number,
        (e->>'num_uniques')::int AS num_uniques,
        (e->>'niceness')::real AS niceness,
        CASE
            WHEN b.id - (e->>'num_uniques')::int <= 2
                THEN (e->>'number')::decimal
            ELSE round(log((e->>'number')::decimal) * 70)
        END AS bucket
    FROM jsonb_array_elements(b.numbers) e
    WHERE (e->>'number')::decimal > 0
) n
ORDER BY b.id, n.num_uniques, n.bucket, n.number
ON CONFLICT DO NOTHING;
