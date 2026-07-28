-- ============================================================
-- 04_validation_comparison_queries.sql
--
-- Comment-level comparison of the two sentiment scoring methods
-- (LLM vs. VADER), used to visualize methodology divergence --
-- e.g. the "VADER misreads sports slang as negative" finding
-- documented in the README's "Sentiment Scoring: VADER vs. LLM"
-- section and in docs/validation_sample_with_llm_v2.csv.
--
-- This is a different kind of query than 03_aggregation_queries.sql:
-- that file aggregates to a DAILY time series for the statistical
-- modeling phase. This file compares LABELS on the SAME INDIVIDUAL
-- COMMENTS across both scoring methods -- it's a validation/
-- methodology artifact, not a modeling input.
--
-- SCOPING NOTE: VADER scored the full ~19,379-comment corpus, but
-- the LLM only scored the ~1,540-comment stratified sample (~50/day),
-- filtered to subject_label = 'about_sga'. This query is therefore
-- restricted to the OVERLAP -- comments that have both an LLM
-- about_sga label AND a VADER label. It is NOT a comparison across
-- the full corpus, since the LLM never scored most of it. Any chart
-- built from this should note that scope explicitly (e.g. a Power BI
-- caption) so it isn't misread as a full-corpus comparison.
-- ============================================================


-- ------------------------------------------------------------
-- View 1: comment-level confusion matrix data -- LLM (about_sga)
-- label vs. VADER label, counted over the overlapping comments
-- ------------------------------------------------------------
CREATE OR REPLACE VIEW llm_vader_label_confusion AS
SELECT
    l.sentiment_label AS llm_label,
    v.sentiment_label AS vader_label,
    COUNT(*)          AS n_comments
FROM sentiment_scores l
JOIN sentiment_scores v
    ON l.comment_id = v.comment_id
    AND v.model_version = 'vader_sentence_filtered_v1'
WHERE l.model_version = 'llm_stratified_v2'
    AND l.subject_label = 'about_sga'
GROUP BY l.sentiment_label, v.sentiment_label
ORDER BY l.sentiment_label, v.sentiment_label;


-- ------------------------------------------------------------
-- Sanity check queries -- run these after creating the view above
-- ------------------------------------------------------------

-- Should return 9 rows (3x3: positive/neutral/negative x positive/neutral/negative)
-- Fewer than 9 just means one label combination had zero comments, not an error --
-- but worth a manual look if so.
-- SELECT COUNT(*) FROM llm_vader_label_confusion;

-- Total comments in the overlap -- tells you how big the comparison sample
-- actually is (expected to be close to the ~1,540-comment stratified sample,
-- since VADER covers the full corpus and should cover ~all of the LLM sample)
-- SELECT SUM(n_comments) FROM llm_vader_label_confusion;

-- Eyeball the full matrix
-- SELECT * FROM llm_vader_label_confusion;

-- Quick look at the diagonal (agreement) vs. off-diagonal (disagreement) split
-- SELECT
--     SUM(CASE WHEN llm_label = vader_label THEN n_comments ELSE 0 END) AS n_agree,
--     SUM(CASE WHEN llm_label != vader_label THEN n_comments ELSE 0 END) AS n_disagree
-- FROM llm_vader_label_confusion;
