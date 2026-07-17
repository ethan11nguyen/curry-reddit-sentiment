"""
sentiment_scoring.py

Scores Curry-related sentiment for each comment in `comments`, writing
results into `sentiment_scores`.

IMPORTANT DESIGN CHOICE: sentence-level filtering, not whole-comment scoring.
Reddit comments frequently mention Curry only in passing while the actual
sentiment is directed at someone else (e.g. "Harden's a ball hog, at least
Curry knows how to play team ball" -- negative sentiment, but about Harden).
Scoring the whole comment would misattribute that sentiment to Curry.

Instead, each comment body is split into sentences, and ONLY the sentences
that actually contain a Curry keyword are kept and scored. This doesn't
solve every case (a single comparative sentence naming two players is still
ambiguous), but it substantially reduces noise from multi-sentence comments
where Curry is just an aside.

See export_validation_sample.py for a companion script that pulls a random
sample for manual labeling, to empirically measure how much of this
misattribution noise remains after sentence-level filtering.

Setup:
    pip install vaderSentiment --break-system-packages
Run:
    python scripts/sentiment_scoring.py
"""

import os
import re
from datetime import datetime, timezone

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

MODEL_VERSION = "vader_sentence_filtered_v1"
BATCH_SIZE = 5000

# Same keyword list used by the loader, kept in sync deliberately.
KEYWORDS = ["curry", "steph", "chef curry", "stephen curry"]

# Standard VADER compound-score thresholds
POSITIVE_THRESHOLD = 0.05
NEGATIVE_THRESHOLD = -0.05

# Naive sentence splitter: break on ., !, ? followed by whitespace.
# Deliberately simple -- Reddit comments don't follow formal prose
# conventions, so a heavier tokenizer (e.g. NLTK punkt) isn't obviously
# better here and adds a data-download dependency.
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

load_dotenv()

PG_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": os.getenv("POSTGRES_PORT", "5432"),
    "dbname": os.getenv("POSTGRES_DB", "curry_sentiment"),
    "user": os.getenv("POSTGRES_USER", "curry_admin"),
    "password": os.getenv("POSTGRES_PASSWORD"),
}


def get_pg_conn():
    return psycopg2.connect(**PG_CONFIG)


def split_sentences(text):
    if not text:
        return []
    # Collapse newlines first so they don't create spurious "sentences"
    text = text.replace("\n", " ").strip()
    if not text:
        return []
    return [s.strip() for s in SENTENCE_SPLIT_RE.split(text) if s.strip()]


def contains_keyword(text):
    lowered = text.lower()
    return any(kw in lowered for kw in KEYWORDS)


def extract_relevant_text(body):
    """
    Returns the subset of `body` actually worth scoring: sentences that
    mention Curry. Falls back to the full body if sentence splitting
    somehow produces no matching sentence (shouldn't normally happen,
    since comments were already keyword-filtered at load time, but a
    keyword could span a sentence boundary in edge cases).
    """
    sentences = split_sentences(body)
    relevant = [s for s in sentences if contains_keyword(s)]
    if not relevant:
        return body  # fallback: score the whole thing
    return " ".join(relevant)


def label_from_compound(compound):
    if compound >= POSITIVE_THRESHOLD:
        return "positive"
    if compound <= NEGATIVE_THRESHOLD:
        return "negative"
    return "neutral"


def fetch_comment_batches(read_conn):
    # Server-side (named) cursor for memory-efficient streaming. This MUST
    # live on its own connection, separate from any connection that commits
    # -- committing invalidates a named cursor mid-iteration.
    with read_conn.cursor(name="comment_stream") as cur:
        cur.itersize = BATCH_SIZE
        cur.execute("SELECT comment_id, body FROM comments WHERE body IS NOT NULL")
        while True:
            rows = cur.fetchmany(BATCH_SIZE)
            if not rows:
                break
            yield rows


def score_batch(analyzer, rows):
    scored_at = datetime.now(timezone.utc)
    results = []
    for comment_id, body in rows:
        relevant_text = extract_relevant_text(body)
        scores = analyzer.polarity_scores(relevant_text)
        compound = scores["compound"]
        label = label_from_compound(compound)
        results.append((comment_id, MODEL_VERSION, compound, label, scored_at))
    return results


def upsert_batch(pg_conn, rows):
    if not rows:
        return
    with pg_conn.cursor() as cur:
        psycopg2.extras.execute_values(
            cur,
            """
            INSERT INTO sentiment_scores
                (comment_id, model_version, sentiment_score, sentiment_label, scored_at)
            VALUES %s
            ON CONFLICT (comment_id, model_version)
            DO UPDATE SET
                sentiment_score = EXCLUDED.sentiment_score,
                sentiment_label = EXCLUDED.sentiment_label,
                scored_at = EXCLUDED.scored_at
            """,
            rows,
        )
    pg_conn.commit()


def main():
    print(f"Model version: {MODEL_VERSION}")
    print("Scoring only Curry-mentioning sentences within each comment (not full comment).\n")

    analyzer = SentimentIntensityAnalyzer()
    read_conn = get_pg_conn()
    write_conn = get_pg_conn()

    total_scored = 0
    total_batches = 0

    try:
        for batch in fetch_comment_batches(read_conn):
            results = score_batch(analyzer, batch)
            upsert_batch(write_conn, results)
            total_batches += 1
            total_scored += len(results)
            print(f"  batch {total_batches}: scored {len(results)} (running total: {total_scored})")
    finally:
        read_conn.close()
        write_conn.close()

    print(f"\nDone. Scored {total_scored} comments across {total_batches} batches.")
    print(f"Results written to sentiment_scores with model_version='{MODEL_VERSION}'.")


if __name__ == "__main__":
    main()