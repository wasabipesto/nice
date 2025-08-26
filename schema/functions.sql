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
