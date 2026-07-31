-- ============================================================
-- 03_aggregation_queries.sql
--
-- Aggregates comment-level sentiment into a daily time series and joins
-- it to SGA's game performance across the full 2024-25 season. This is
-- the bridge between the raw scored data (comments + sentiment_scores +
-- player_stats) and the statistical modeling phase (OLS / ARMA / event
-- study).
--
-- SCOPING CHANGE FROM THE ORIGINAL CURRY-ERA VERSION OF THIS FILE:
-- back then, VADER scored the FULL ~19K-comment corpus (unfiltered),
-- while the LLM only covered a smaller stratified sample -- two
-- differently-sized populations. This time, BOTH VADER and the LLM were
-- deliberately run against the exact same stratified_sample population
-- (~23,093 comments, see stratified_sample_comments.py for how that was
-- built: 50/day baseline across all 273 days + 150/day boost on the 99
-- actual game days, all restricted to comments genuinely mentioning
-- SGA). This means there is no "full corpus" VADER series in this
-- pipeline -- both daily series below draw from the identical
-- stratified_sample population, differing only in which model scored
-- them. This was a deliberate tradeoff: it sacrifices VADER's
-- full-corpus coverage in exchange for a clean, directly comparable
-- VADER-vs-LLM measurement (see sql/04_validation_comparison_queries.sql).
--
-- Two parallel sentiment series are built, NOT averaged together:
--   1. VADER, sentence-filtered, drawn from stratified_sample
--      (~23,093 comments) -- no subject filtering applied, since VADER
--      has no subject-classification ability at all.
--   2. LLM, drawn from the same stratified_sample, filtered to
--      subject_label = 'about_sga' only (~13,389 of the 23,093 --
--      see the full breakdown from llm_stratified_scoring.py's run).
--      This is the primary series -- validated against manual labels
--      at 94.7% accuracy on the 150-comment validation set.
--
-- Every day of the 2024-25 season (Oct 1, 2024 - June 30, 2025, 273
-- days) gets a sentiment average. Only ~99 of those days have an SGA
-- game -- the LEFT JOIN to player_stats intentionally leaves non-game
-- days with NULL performance columns, which is expected, not a data
-- error.
--
-- EVENT MARKER: SGA's 2024-25 MVP was announced May 21, 2025 (also
-- Finals MVP after the championship run) -- the post_mvp_announcement
-- flag below marks days on/after that date, analogous to the
-- post_mvp_announcement flag in the original Curry pipeline (May 4,
-- 2015 there).
-- ============================================================


-- ------------------------------------------------------------
-- View 1: daily VADER sentiment, drawn from stratified_sample
-- (no subject filtering -- VADER can't classify subject at all)
-- ------------------------------------------------------------
CREATE OR REPLACE VIEW daily_sentiment_vader AS
SELECT
    DATE(c.created_at)         AS comment_date,
    COUNT(*)                   AS n_comments,
    AVG(s.sentiment_score)     AS avg_sentiment_score,
    SUM(CASE WHEN s.sentiment_label = 'positive' THEN 1 ELSE 0 END) AS n_positive,
    SUM(CASE WHEN s.sentiment_label = 'negative' THEN 1 ELSE 0 END) AS n_negative,
    SUM(CASE WHEN s.sentiment_label = 'neutral'  THEN 1 ELSE 0 END) AS n_neutral
FROM comments c
JOIN stratified_sample ss ON ss.comment_id = c.comment_id
JOIN sentiment_scores s
    ON c.comment_id = s.comment_id AND s.model_version = 'vader_sentence_filtered_v1'
WHERE c.created_at IS NOT NULL
GROUP BY DATE(c.created_at)
ORDER BY comment_date;


-- ------------------------------------------------------------
-- View 2: daily LLM sentiment, about_sga only (primary series)
-- ------------------------------------------------------------
CREATE OR REPLACE VIEW daily_sentiment_llm AS
SELECT
    DATE(c.created_at)         AS comment_date,
    COUNT(*)                   AS n_comments,
    AVG(s.sentiment_score)     AS avg_sentiment_score,
    SUM(CASE WHEN s.sentiment_label = 'positive' THEN 1 ELSE 0 END) AS n_positive,
    SUM(CASE WHEN s.sentiment_label = 'negative' THEN 1 ELSE 0 END) AS n_negative,
    SUM(CASE WHEN s.sentiment_label = 'neutral'  THEN 1 ELSE 0 END) AS n_neutral
FROM comments c
JOIN stratified_sample ss ON ss.comment_id = c.comment_id
JOIN sentiment_scores s
    ON c.comment_id = s.comment_id AND s.model_version = 'llm_stratified_v1'
WHERE c.created_at IS NOT NULL
    AND s.subject_label = 'about_sga'
GROUP BY DATE(c.created_at)
ORDER BY comment_date;


-- ------------------------------------------------------------
-- View 3: daily game performance (one row per game date; SGA played
-- at most one game per day in this window)
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
-- statistical modeling phase. One row per day in the 2024-25 season.
-- ------------------------------------------------------------
CREATE OR REPLACE VIEW daily_sentiment_and_performance AS
SELECT
    d.day AS comment_date,

    l.n_comments          AS llm_n_comments,
    l.avg_sentiment_score  AS llm_avg_score,
    l.n_positive           AS llm_n_positive,
    l.n_negative           AS llm_n_negative,
    l.n_neutral            AS llm_n_neutral,

    v.n_comments          AS vader_n_comments,
    v.avg_sentiment_score  AS vader_avg_score,
    v.n_positive           AS vader_n_positive,
    v.n_negative           AS vader_n_negative,
    v.n_neutral            AS vader_n_neutral,

    p.matchup, p.opponent, p.home_away, p.win_loss,
    p.points, p.rebounds, p.assists, p.steals, p.blocks, p.turnovers,
    p.fg_pct, p.fg3_pct, p.ft_pct, p.plus_minus,
    (p.game_date IS NOT NULL) AS is_game_day,

    -- flag for the event study: on/after the May 21, 2025 MVP announcement
    (d.day >= DATE '2025-05-21') AS post_mvp_announcement

FROM (
    -- generate every calendar day in the 2024-25 season window, so days
    -- with zero sampled comments or no game still appear as a row rather
    -- than silently vanishing
    SELECT generate_series(DATE '2024-10-01', DATE '2025-06-30', INTERVAL '1 day')::date AS day
) d
LEFT JOIN daily_sentiment_vader v ON v.comment_date = d.day
LEFT JOIN daily_sentiment_llm l ON l.comment_date = d.day
LEFT JOIN daily_player_performance p ON p.game_date = d.day
ORDER BY d.day;


-- ------------------------------------------------------------
-- Sanity check queries -- run these after creating the views above
-- ------------------------------------------------------------

-- Should return 273 rows (one per day, Oct 1 2024 - June 30 2025)
-- SELECT COUNT(*) FROM daily_sentiment_and_performance;

-- Should show ~99 game days with non-null performance columns
-- SELECT COUNT(*) FROM daily_sentiment_and_performance WHERE is_game_day;

-- Eyeball the full table
-- SELECT * FROM daily_sentiment_and_performance;