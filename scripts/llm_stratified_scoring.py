"""
llm_stratified_scoring.py

Scores Curry sentiment using an LLM (via Hugging Face Inference Providers),
on a STRATIFIED DAILY SAMPLE of comments rather than the full ~19k -- caps
comments scored per calendar day so every day of May 2015 is represented
in the resulting time series, without the cost/time of scoring everything.

WHY STRATIFIED (not a flat random sample): comment volume varies a lot
day to day (144 on the quietest day, 2,018 on the busiest -- see the
per-day count query used to check this before building the sampler). A
flat random sample risks leaving light-volume days with too few comments
to compute a meaningful daily sentiment average. Capping per-day ensures
every day clears a reasonable minimum, without over-sampling already
comment-heavy days.

This becomes the PRIMARY sentiment time series for the statistical
modeling (OLS / ARMA / event study) -- VADER's full-corpus score
(sentiment_scoring.py) is kept as a secondary comparison method, since the
validation sample showed it struggles with subject attribution and sports
slang.

Prerequisite (run once, see script docstring notes):
    ALTER TABLE sentiment_scores ADD COLUMN subject_label VARCHAR(20);
    (also update sql/01_reddit_schema.sql to match)

Setup:
    pip install huggingface_hub --break-system-packages
    Add to .env:  HF_TOKEN=hf_your_token_here

Run:
    python scripts/llm_stratified_scoring.py
"""

import json
import os
import re
import time
from datetime import datetime, timezone

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

MODEL_VERSION = "llm_stratified_v1"
MODEL = "meta-llama/Llama-3.1-8B-Instruct"
PROVIDER = "auto"

# Cap per calendar day. Every day in the dataset has >=144 comments, so at
# 50/day this samples ~1,550 comments total (31 days x up to 50) rather
# than all 19k -- adjust up/down based on how the first run goes.
SAMPLE_PER_DAY = 50

REQUEST_DELAY_SECONDS = 1.5
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 5

SYSTEM_PROMPT = """You analyze Reddit comments from r/nba for sentiment specifically about \
the basketball player Stephen Curry. Many comments mention Curry only in passing \
while actually expressing sentiment about someone or something else (e.g. \
comparing him to another player, or using him as a reference point). Your job \
is to identify whether the comment expresses genuine sentiment TOWARD Curry \
himself, and if so, what that sentiment is.

Respond with ONLY a JSON object, no other text, in this exact format:
{"subject": "about_curry" | "incidental" | "comparative" | "unclear", \
"sentiment": "positive" | "negative" | "neutral", "score": <float from -1.0 to 1.0>}

Field meanings:
- "subject": "about_curry" if the sentiment is genuinely directed at Curry; \
"incidental" if Curry is mentioned but the sentiment is about someone/something \
else; "comparative" if it's a direct comparison where sentiment is split between \
Curry and another player; "unclear" if you can't tell or there's no real sentiment.
- "sentiment": the overall sentiment label. If subject is "incidental", this \
should reflect sentiment toward Curry specifically (often "neutral" if none \
is actually expressed toward him).
- "score": -1.0 (very negative) to 1.0 (very positive), 0.0 = neutral. \
This should reflect sentiment toward Curry specifically, not the whole comment.

Respond with the JSON object ONLY. No explanation, no examples, no extra text \
before or after it. Just the single JSON object.
"""

load_dotenv()

PG_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": os.getenv("POSTGRES_PORT", "5432"),
    "dbname": os.getenv("POSTGRES_DB", "curry_sentiment"),
    "user": os.getenv("POSTGRES_USER", "curry_admin"),
    "password": os.getenv("POSTGRES_PASSWORD"),
}

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise SystemExit("ERROR: HF_TOKEN not found in .env. See script docstring for setup.")


def get_pg_conn():
    return psycopg2.connect(**PG_CONFIG)


def get_hf_client():
    return InferenceClient(api_key=HF_TOKEN, provider=PROVIDER)


def fetch_distinct_days(pg_conn):
    with pg_conn.cursor() as cur:
        cur.execute(
            """
            SELECT DISTINCT DATE(created_utc) AS d
            FROM comments
            WHERE created_utc IS NOT NULL
            ORDER BY d
            """
        )
        return [row[0] for row in cur.fetchall()]


def fetch_scored_count_for_day(pg_conn, day):
    """How many comments on this day already have a score for MODEL_VERSION."""
    with pg_conn.cursor() as cur:
        cur.execute(
            """
            SELECT COUNT(DISTINCT c.comment_id)
            FROM comments c
            JOIN sentiment_scores s
                ON c.comment_id = s.comment_id AND s.model_version = %s
            WHERE DATE(c.created_utc) = %s
            """,
            (MODEL_VERSION, day),
        )
        return cur.fetchone()[0]


def fetch_sample_for_day(pg_conn, day, limit):
    # LEFT JOIN + IS NULL excludes comments already scored with this
    # model_version -- this is what makes re-running the script after an
    # interruption (e.g. hitting a billing cap) safe: already-scored
    # comments are automatically skipped rather than re-sampled/re-billed.
    with pg_conn.cursor() as cur:
        cur.execute(
            """
            SELECT c.comment_id, c.body
            FROM comments c
            LEFT JOIN sentiment_scores s
                ON c.comment_id = s.comment_id AND s.model_version = %s
            WHERE DATE(c.created_utc) = %s
                AND c.body IS NOT NULL
                AND s.comment_id IS NULL
            ORDER BY random()
            LIMIT %s
            """,
            (MODEL_VERSION, day, limit),
        )
        return cur.fetchall()


def score_comment(client, body):
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            completion = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": body},
                ],
                temperature=0.1,
                max_tokens=150,
            )
            raw = completion.choices[0].message.content.strip()
            cleaned = raw.strip("`").removeprefix("json").strip()
            try:
                parsed = json.loads(cleaned)
            except json.JSONDecodeError:
                match = re.search(r"\{.*?\}", raw, re.DOTALL)
                if not match:
                    raise
                parsed = json.loads(match.group(0))
            return parsed.get("subject"), parsed.get("sentiment"), parsed.get("score")
        except json.JSONDecodeError as e:
            last_error = f"JSON parse failed on response: {raw!r} ({e})"
        except Exception as e:
            last_error = str(e)

        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY_SECONDS)

    print(f"    giving up after {MAX_RETRIES} attempts: {last_error}")
    return None, None, None


def upsert_score(pg_conn, comment_id, subject, sentiment, score):
    scored_at = datetime.now(timezone.utc)
    with pg_conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO sentiment_scores
                (comment_id, model_version, sentiment_score, sentiment_label, subject_label, scored_at)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (comment_id, model_version)
            DO UPDATE SET
                sentiment_score = EXCLUDED.sentiment_score,
                sentiment_label = EXCLUDED.sentiment_label,
                subject_label = EXCLUDED.subject_label,
                scored_at = EXCLUDED.scored_at
            """,
            (comment_id, MODEL_VERSION, score, sentiment, subject, scored_at),
        )
    pg_conn.commit()


def main():
    print(f"Model: {MODEL} (provider={PROVIDER})")
    print(f"Sample cap: {SAMPLE_PER_DAY} comments/day\n")

    read_conn = get_pg_conn()
    write_conn = get_pg_conn()
    hf_client = get_hf_client()

    try:
        days = fetch_distinct_days(read_conn)
        print(f"Found {len(days)} distinct days in the dataset.\n")

        total_scored = 0
        total_failed = 0
        total_skipped_days = 0

        for day in days:
            already = fetch_scored_count_for_day(read_conn, day)
            remaining = SAMPLE_PER_DAY - already

            if remaining <= 0:
                print(f"{day}: already has {already}/{SAMPLE_PER_DAY} scored -- skipping")
                total_skipped_days += 1
                continue

            rows = fetch_sample_for_day(read_conn, day, remaining)
            print(f"{day}: {already} already scored, sampling {len(rows)} more (target {SAMPLE_PER_DAY})")

            for comment_id, body in rows:
                subject, sentiment, score = score_comment(hf_client, body)
                if subject is None:
                    total_failed += 1
                else:
                    upsert_score(write_conn, comment_id, subject, sentiment, score)
                    total_scored += 1
                time.sleep(REQUEST_DELAY_SECONDS)

            print(f"  running total: {total_scored} scored, {total_failed} failed\n")

    finally:
        read_conn.close()
        write_conn.close()

    print(f"Done. Scored {total_scored} new, {total_skipped_days} days already complete, {total_failed} failed.")
    print(f"Results written to sentiment_scores with model_version='{MODEL_VERSION}'.")


if __name__ == "__main__":
    main()