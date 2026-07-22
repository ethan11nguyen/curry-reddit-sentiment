-- ============================================================
-- 03_aggregation_queries.sql
--
-- Aggregates comment-level sentiment into a daily time series and joins
-- it to Curry's game performance. This is the bridge between the raw
-- scored data (comments + sentiment_scores + player_stats) and the
-- statistical modeling phase (OLS / ARMA / event study).
--
-- Two parallel sentiment series are built, NOT averaged together:
--   1. VADER, full corpus (~19,379 comments) -- no subject filtering
--      available, since VADER never produced a subject classification.
--      This is the noisier, less trustworthy series (see validation
--      writeup: VADER conflates incidental/comparative mentions with
--      genuine Curry sentiment).
--   2. LLM, stratified sample (~50/day, ~1,540 total), filtered to
--      subject_label = 'about_curry' only. This is the primary series --
--      validated against manual labels at ~78% binary accuracy for the
--      about_curry classification specifically.
--
-- Every day of May 2015 gets a sentiment average. Only ~21 of those days
-- have a Curry game -- the LEFT JOIN to player_stats intentionally leaves
-- non-game days with NULL performance columns, which is expected, not a
-- data error.
-- ============================================================


-- ------------------------------------------------------------
-- View 1: daily VADER sentiment (full corpus, unfiltered)
-- ------------------------------------------------------------
CREATE OR REPLACE VIEW daily_sentiment_vader AS
SELECT
    DATE(c.created_utc)        AS comment_date,
    COUNT(*)                   AS n_comments,
    AVG(s.sentiment_score)     AS avg_sentiment_score,
    SUM(CASE WHEN s.sentiment_label = 'positive' THEN 1 ELSE 0 END) AS n_positive,
    SUM(CASE WHEN s.sentiment_label = 'negative' THEN 1 ELSE 0 END) AS n_negative,
    SUM(CASE WHEN s.sentiment_label = 'neutral'  THEN 1 ELSE 0 END) AS n_neutral
FROM comments c
JOIN sentiment_scores s
    ON c.comment_id = s.comment_id AND s.model_version = 'vader_sentence_filtered_v1'
WHERE c.created_utc IS NOT NULL
GROUP BY DATE(c.created_utc)
ORDER BY comment_date;


-- ------------------------------------------------------------
-- View 2: daily LLM sentiment, about_curry only (primary series)
-- ------------------------------------------------------------
CREATE OR REPLACE VIEW daily_sentiment_llm AS
SELECT
    DATE(c.created_utc)        AS comment_date,
    COUNT(*)                   AS n_comments,
    AVG(s.sentiment_score)     AS avg_sentiment_score,
    SUM(CASE WHEN s.sentiment_label = 'positive' THEN 1 ELSE 0 END) AS n_positive,
    SUM(CASE WHEN s.sentiment_label = 'negative' THEN 1 ELSE 0 END) AS n_negative,
    SUM(CASE WHEN s.sentiment_label = 'neutral'  THEN 1 ELSE 0 END) AS n_neutral
FROM comments c
JOIN sentiment_scores s
    ON c.comment_id = s.comment_id AND s.model_version = 'llm_stratified_v2'
WHERE c.created_utc IS NOT NULL
    AND s.subject_label = 'about_curry'
GROUP BY DATE(c.created_utc)
ORDER BY comment_date;


-- ------------------------------------------------------------
-- View 3: daily game performance (one row per game date;
-- Curry played at most one game per day in this window)
-- ------------------------------------------------------------
CREATE OR REPLACE VIEW daily_player_performance AS
SELECT
    game_date,
    matchup,
    opponent,
    home_away,
    win_loss,
    points,
    rebounds,
    assists,
    steals,
    blocks,
    turnovers,
    fg_pct,
    fg3_pct,
    ft_pct,
    plus_minus
FROM player_stats
ORDER BY game_date;


-- ------------------------------------------------------------
-- View 4: the combined daily table -- this is what feeds the
-- statistical modeling phase. One row per day in May 2015.
-- ------------------------------------------------------------
CREATE OR REPLACE VIEW daily_sentiment_and_performance AS
SELECT
    d.day AS comment_date,

    l.n_comments        AS llm_n_comments,
    l.avg_sentiment_score AS llm_avg_score,
    l.n_positive         AS llm_n_positive,
    l.n_negative         AS llm_n_negative,
    l.n_neutral          AS llm_n_neutral,

    v.n_comments        AS vader_n_comments,
    v.avg_sentiment_score AS vader_avg_score,
    v.n_positive         AS vader_n_positive,
    v.n_negative         AS vader_n_negative,
    v.n_neutral          AS vader_n_neutral,

    p.matchup, p.opponent, p.home_away, p.win_loss,
    p.points, p.rebounds, p.assists, p.steals, p.blocks, p.turnovers,
    p.fg_pct, p.fg3_pct, p.ft_pct, p.plus_minus,
    (p.game_date IS NOT NULL) AS is_game_day,

    -- flag for the event study: on/after the May 4, 2015 MVP announcement
    (d.day >= DATE '2015-05-04') AS post_mvp_announcement

FROM (
    -- generate every calendar day in May 2015, so days with zero comments
    -- or no game still appear as a row rather than silently vanishing
    SELECT generate_series(DATE '2015-05-01', DATE '2015-05-31', INTERVAL '1 day')::date AS day
) d
LEFT JOIN daily_sentiment_vader v ON v.comment_date = d.day
LEFT JOIN daily_sentiment_llm l ON l.comment_date = d.day
LEFT JOIN daily_player_performance p ON p.game_date = d.day
ORDER BY d.day;


-- ------------------------------------------------------------
-- Sanity check queries -- run these after creating the views above
-- ------------------------------------------------------------

-- Should return 31 rows (one per day in May)
-- SELECT COUNT(*) FROM daily_sentiment_and_performance;

-- Should show ~21 game days with non-null performance columns
-- SELECT COUNT(*) FROM daily_sentiment_and_performance WHERE is_game_day;

-- Eyeball the full table
-- SELECT * FROM daily_sentiment_and_performance;