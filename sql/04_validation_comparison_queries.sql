-- ============================================================
-- 04_validation_comparison_queries.sql
--
-- Comment-level comparison of the two sentiment scoring methods
-- (LLM vs. VADER) for the SGA / 2024-25 season pipeline.
--
-- KEY DIFFERENCE FROM THE ORIGINAL CURRY-ERA VERSION OF THIS FILE:
-- back then, VADER covered the full ~19K-comment corpus while the LLM
-- only covered a stratified sample filtered to subject_label =
-- 'about_curry' -- meaning the confusion matrix was necessarily
-- restricted to that "best case" overlap, and likely understated real
-- disagreement between the methods (see that file's own scoping note).
--
-- This time, BOTH VADER (vader_sentiment_scoring.py) and the LLM
-- (llm_stratified_scoring.py) were deliberately run against the exact
-- same population -- the full stratified_sample table (~23,093
-- comments) -- so this file can build a genuinely unrestricted,
-- apples-to-apples comparison across the whole sample, not just the
-- subset the LLM was confident was about_sga.
--
-- Two views below:
--   1. Unrestricted: every comment in the sample, VADER's sentiment
--      label vs. LLM's sentiment label, regardless of subject
--      classification.
--   2. about_sga-restricted: same comparison, but filtered to only
--      comments the LLM classified as genuinely about SGA -- the
--      "best case" comparison, analogous to what the old Curry pipeline
--      measured (though that one was restricted by necessity, not choice).
--
-- NOTE ON THE MANUAL/THIRD-WAY COMPARISON: the 150-row manually-labeled
-- validation set lives in docs/validation_sample.csv, not in Postgres,
-- so a three-way (manual vs. VADER vs. LLM) comparison is done via
-- scripts/check_llm_accuracy.py (Python, reads the CSVs directly) rather
-- than SQL here. These two SQL views are for the full ~23K-comment
-- sample, which has no manual labels to compare against -- they show
-- LLM-vs-VADER agreement/disagreement at full scale, not accuracy
-- against ground truth (that's what the 94.7% figure from the 150-row
-- validation run already established).
-- ============================================================


-- ------------------------------------------------------------
-- View 1: unrestricted VADER vs LLM sentiment_label confusion matrix,
-- across the full stratified_sample (~23,093 comments)
--
-- DISPLAY LABELS: renamed here (rather than left as raw lowercase
-- values) so the Power BI matrix visual shows readable row/column
-- headers directly, without needing per-visual renaming in Power BI
-- itself. The LLM occasionally (8 of 23,093 rows) returned an
-- off-schema "mixed" sentiment_label instead of the three defined in
-- the prompt (positive/negative/neutral) -- folded into "Neutral" here
-- for display purposes, since it's a rare, undefined value that would
-- otherwise show as an unexplained 4th category in the matrix. This is
-- a presentation-layer choice only; the raw "mixed" value is untouched
-- in sentiment_scores itself for anyone querying the raw table directly.
-- ------------------------------------------------------------
CREATE OR REPLACE VIEW llm_vader_confusion_unrestricted AS
SELECT
    CASE l.sentiment_label
        WHEN 'negative' THEN 'LLM: Negative'
        WHEN 'positive' THEN 'LLM: Positive'
        ELSE 'LLM: Neutral'  -- covers 'neutral' and the rare off-schema 'mixed'
    END AS llm_label,
    CASE v.sentiment_label
        WHEN 'negative' THEN 'VADER: Negative'
        WHEN 'positive' THEN 'VADER: Positive'
        ELSE 'VADER: Neutral'
    END AS vader_label,
    COUNT(*)          AS n_comments
FROM sentiment_scores l
JOIN sentiment_scores v
    ON l.comment_id = v.comment_id
    AND v.model_version = 'vader_sentence_filtered_v1'
WHERE l.model_version = 'llm_stratified_v1'
GROUP BY 1, 2
ORDER BY 1, 2;


-- ------------------------------------------------------------
-- View 2: same comparison, restricted to comments the LLM classified
-- as about_sga specifically -- the "best case" / most-comparable subset
-- ------------------------------------------------------------
CREATE OR REPLACE VIEW llm_vader_confusion_about_sga AS
SELECT
    CASE l.sentiment_label
        WHEN 'negative' THEN 'LLM: Negative'
        WHEN 'positive' THEN 'LLM: Positive'
        ELSE 'LLM: Neutral'  -- covers 'neutral' and the rare off-schema 'mixed'
    END AS llm_label,
    CASE v.sentiment_label
        WHEN 'negative' THEN 'VADER: Negative'
        WHEN 'positive' THEN 'VADER: Positive'
        ELSE 'VADER: Neutral'
    END AS vader_label,
    COUNT(*)          AS n_comments
FROM sentiment_scores l
JOIN sentiment_scores v
    ON l.comment_id = v.comment_id
    AND v.model_version = 'vader_sentence_filtered_v1'
WHERE l.model_version = 'llm_stratified_v1'
    AND l.subject_label = 'about_sga'
GROUP BY 1, 2
ORDER BY 1, 2;


-- ------------------------------------------------------------
-- Sanity check queries -- run these after creating the views above
-- ------------------------------------------------------------

-- Should sum to 23,093 (the full stratified_sample size)
-- SELECT SUM(n_comments) FROM llm_vader_confusion_unrestricted;

-- Should sum to however many rows LLM classified as about_sga (~13,389)
-- SELECT SUM(n_comments) FROM llm_vader_confusion_about_sga;

-- Quick agree/disagree split, unrestricted view
-- SELECT
--     SUM(CASE WHEN llm_label = vader_label THEN n_comments ELSE 0 END) AS n_agree,
--     SUM(CASE WHEN llm_label != vader_label THEN n_comments ELSE 0 END) AS n_disagree
-- FROM llm_vader_confusion_unrestricted;

-- Same split, about_sga-restricted view -- compare this rate against
-- the unrestricted one above; if about_sga-restricted agreement is
-- meaningfully higher, that's evidence VADER does relatively better
-- when the comment actually is about SGA
-- SELECT
--     SUM(CASE WHEN llm_label = vader_label THEN n_comments ELSE 0 END) AS n_agree,
--     SUM(CASE WHEN llm_label != vader_label THEN n_comments ELSE 0 END) AS n_disagree
-- FROM llm_vader_confusion_about_sga;

-- Eyeball the full matrices
-- SELECT * FROM llm_vader_confusion_unrestricted;
-- SELECT * FROM llm_vader_confusion_about_sga;