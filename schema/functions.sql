-- Total amount completed per day for the last 30 days
SELECT
    DATE(s.submit_time) as date,
    COUNT(s) as num_submissions,
    SUM(f.range_size) as total_range
FROM submissions s
JOIN fields f ON s.field_id = f.id
WHERE
    s.submit_time >= CURRENT_DATE - INTERVAL '30 days'
    AND s.disqualified = false
    AND s.search_mode = 'detailed'
GROUP BY DATE(s.submit_time)
ORDER BY date DESC;

-- Leaderboard for submissions by user for the last 30 days
SELECT
    s.username,
    COUNT(s) as num_submissions,
    SUM(f.range_size) as total_range
FROM submissions s
JOIN fields f ON s.field_id = f.id
WHERE
    s.submit_time >= CURRENT_DATE - INTERVAL '30 days'
    AND s.disqualified = false
GROUP BY s.username
ORDER BY total_range DESC;

-- Leaderboard for fastest submissions in the last 30 days
SELECT
    s.username,
    (s.submit_time - c.claim_time) as duration,
    s.search_mode,
    ROUND(f.range_size / EXTRACT(EPOCH FROM (s.submit_time - c.claim_time)) / 1000000, 2) as speed_e6
FROM submissions s
JOIN claims c ON s.claim_id = c.id
JOIN fields f ON s.field_id = f.id
WHERE
    s.submit_time >= CURRENT_DATE - INTERVAL '30 days'
    AND s.disqualified = false
ORDER BY speed_e6 DESC
LIMIT 10;
