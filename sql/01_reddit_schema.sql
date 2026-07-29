-- ============================================================
-- 01_reddit_schema.sql
--
-- Schema for pushshift-derived r/nba comments (Oct 2024 - June 2025
-- season). Source: monthly RC_YYYY-MM.zst dumps from Academic Torrents,
-- filtered to subreddit == "nba" via scripts/filter_subreddit_streaming.py.
--
-- Pushshift comment objects contain ~50 fields, most irrelevant to
-- sentiment analysis (author flair richtext, awardings, mod fields,
-- etc.). This schema captures the fields actually needed as typed
-- columns, and stores everything else in a JSONB column so no data
-- is lost, without needing to model every field individually.
-- ============================================================

CREATE TABLE IF NOT EXISTS comments (
    comment_id          TEXT PRIMARY KEY,        -- pushshift "id"
    author               TEXT,
    body                 TEXT NOT NULL,
    subreddit            TEXT NOT NULL,           -- should always be "nba" post-filter
    link_id              TEXT,                    -- submission/post this comment belongs to
    parent_id            TEXT,                    -- immediate parent (comment or post)
    permalink            TEXT,
    score                INTEGER,
    controversiality     INTEGER,
    created_utc          BIGINT NOT NULL,         -- raw unix epoch, as pushshift stores it
    created_at           TIMESTAMPTZ GENERATED ALWAYS AS (
                             to_timestamp(created_utc)
                         ) STORED,                 -- computed real timestamp, for date filtering/joins
    retrieved_on         BIGINT,
    raw_json             JSONB,                    -- full original object, for anything not modeled above

    inserted_at          TIMESTAMPTZ DEFAULT now()
);

-- Indexes -- important at this scale (7.18M rows, vs. ~19K in the old
-- Curry pipeline). Without these, joins/filters in the aggregation and
-- validation-comparison queries would degrade noticeably.
CREATE INDEX IF NOT EXISTS idx_comments_created_at ON comments (created_at);
CREATE INDEX IF NOT EXISTS idx_comments_link_id ON comments (link_id);



-- ------------------------------------------------------------
-- sentiment_scores: holds VADER and LLM sentiment output, one row
-- per (comment_id, model_version) pair. subject_label is only
-- populated by the LLM (VADER has no subject-classification ability).
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS sentiment_scores (
    comment_id       TEXT NOT NULL REFERENCES comments(comment_id),
    model_version    TEXT NOT NULL,     -- e.g. 'vader_sentence_filtered_v1', 'llm_stratified_v1'
    sentiment_score  NUMERIC,
    sentiment_label  TEXT,              -- 'positive' / 'negative' / 'neutral'
    subject_label    TEXT,              -- 'about_sga' / 'incidental' / 'comparative' / 'unclear' (LLM only)
    scored_at        TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (comment_id, model_version)
);

CREATE INDEX IF NOT EXISTS idx_sentiment_scores_model_version ON sentiment_scores (model_version);














