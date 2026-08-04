-- ============================================================
-- 05_powerbi_views.sql
--
-- Additional views built specifically to feed the Power BI dashboard,
-- separate from 03_aggregation_queries.sql's core modeling views.
--
-- KEY VIEW: daily_sentiment_and_performance_next_day -- adds a next-day
-- sentiment column via a self-join (LEFT JOIN on comment_date + 1 day)
-- against daily_sentiment_and_performance. This mirrors EXACTLY the
-- date-shift logic in scripts/statistical_modeling.py's run_ols_next_day()
-- function (pandas: game_days["next_day"] = comment_date + Timedelta(1)),
-- so the dashboard's numbers match the validated Python analysis rather
-- than reimplementing the shift separately in DAX/Power Query, which
-- would risk silently drifting out of sync with the actual statistical
-- results reported in docs/findings.md.
-- ============================================================

CREATE OR REPLACE VIEW daily_sentiment_and_performance_next_day AS
SELECT
    d.comment_date,
    d.llm_avg_score          AS same_day_llm_avg_score,
    d.llm_n_comments         AS same_day_llm_n_comments,
    nd.llm_avg_score         AS next_day_llm_avg_score,
    nd.llm_n_comments        AS next_day_llm_n_comments,

    d.matchup, d.opponent, d.home_away, d.win_loss,
    d.points, d.rebounds, d.assists, d.steals, d.blocks, d.turnovers,
    d.fg_pct, d.fg3_pct, d.ft_pct, d.plus_minus,
    d.is_game_day,
    d.post_mvp_announcement

FROM daily_sentiment_and_performance d
LEFT JOIN daily_sentiment_and_performance nd
    ON nd.comment_date = d.comment_date + INTERVAL '1 day'
ORDER BY d.comment_date;


-- ------------------------------------------------------------
-- Sanity check -- should return 99 rows with non-null next_day_llm_avg_score
-- (matches Python's "99/99 game days with a next-day sentiment value
-- available" from the actual statistical_modeling.py run)
-- ------------------------------------------------------------
-- SELECT COUNT(*) FROM daily_sentiment_and_performance_next_day
-- WHERE is_game_day AND next_day_llm_avg_score IS NOT NULL;
