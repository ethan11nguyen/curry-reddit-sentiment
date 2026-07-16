-- ============================================================
-- Reddit data schema: posts, comments, sentiment_scores
-- ============================================================

CREATE TABLE IF NOT EXISTS posts (
    post_id         VARCHAR(20) PRIMARY KEY,   -- Reddit's base36 post ID (e.g. 't3_abc123' -> 'abc123')
    title           TEXT NOT NULL,
    selftext        TEXT,
    author          VARCHAR(100),
    created_utc     TIMESTAMP, -- nullable bc kaggle sourced placeholder posts have no real metadata
    score           INTEGER,
    upvote_ratio    NUMERIC(4,3),
    num_comments    INTEGER,
    url             TEXT,
    flair           VARCHAR(100),
    scraped_at      TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS comments (
    comment_id      VARCHAR(20) PRIMARY KEY,   -- Reddit's base36 comment ID
    post_id         VARCHAR(20) NOT NULL REFERENCES posts(post_id) ON DELETE CASCADE,
    parent_id       VARCHAR(20),               -- either a post_id or another comment_id
    body            TEXT NOT NULL,
    author          VARCHAR(100),
    created_utc     TIMESTAMP NOT NULL,
    score           INTEGER,
    is_submitter    BOOLEAN DEFAULT FALSE,
    scraped_at      TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Separate from `comments` on purpose: lets you re-score with a new model
-- version without touching raw data, and keep every scoring run's history.
CREATE TABLE IF NOT EXISTS sentiment_scores (
    id              SERIAL PRIMARY KEY,
    comment_id      VARCHAR(20) NOT NULL REFERENCES comments(comment_id) ON DELETE CASCADE,
    model_version   VARCHAR(50) NOT NULL,      -- e.g. 'vader_v1', 'textblob_v1', 'roberta_v1'
    sentiment_score NUMERIC(6,4),               -- raw compound/polarity score, NULL until scored
    sentiment_label VARCHAR(10),                -- 'positive' | 'neutral' | 'negative', NULL until scored
    scored_at       TIMESTAMP,
    UNIQUE (comment_id, model_version)          -- one score per comment per model version
);

-- Indexes for the queries you'll actually run: time-based aggregation and joins
CREATE INDEX IF NOT EXISTS idx_posts_created_utc ON posts(created_utc);
CREATE INDEX IF NOT EXISTS idx_comments_created_utc ON comments(created_utc);
CREATE INDEX IF NOT EXISTS idx_comments_post_id ON comments(post_id);
CREATE INDEX IF NOT EXISTS idx_sentiment_comment_id ON sentiment_scores(comment_id);
CREATE INDEX IF NOT EXISTS idx_sentiment_label ON sentiment_scores(sentiment_label);
