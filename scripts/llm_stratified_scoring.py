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

MODEL_VERSION = "llm_stratified_v2"
MODEL = "meta-llama/Llama-3.1-8B-Instruct"
PROVIDER = "auto"

# Cap per calendar day. Every day in the dataset has >=144 comments, so at
# 50/day this samples ~1,550 comments total (31 days x up to 50) rather
# than all 19k -- adjust up/down based on how the first run goes.
SAMPLE_PER_DAY = 50

REQUEST_DELAY_SECONDS = 1.5
MAX_RETRIES = 3
API_ERROR_RETRY_DELAY_SECONDS = 5
FORMAT_RETRY_DELAY_SECONDS = 1

REFUSAL_PATTERNS = [
    "i cannot create", "i can't create", "i cannot generate", "i can't generate",
    "i cannot help with", "i can't help with", "i'm not able to", "i am not able to",
    "i cannot provide", "i can't provide",
]


def looks_like_refusal(raw):
    lowered = raw.lower()
    return any(pattern in lowered for pattern in REFUSAL_PATTERNS)

SYSTEM_PROMPT = """You analyze Reddit comments from r/nba for sentiment specifically about \
the basketball player Stephen Curry.

CATEGORIES:
- "about_curry": the comment expresses genuine sentiment directed at Curry himself. \
This INCLUDES cases where a comparison to another player is used merely as \
rhetorical framing, as long as the substantive point is about Curry. \
Example: "Curry could have gotten the charge, I don't know why he didn't fall \
over" -> about_curry (negative, mild criticism of a specific Curry play). \
Example: "CP was off of a hammy though, Steph is 100% healthy" -> about_curry \
(positive -- states Curry's health status directly).
- "incidental": Curry's name appears but the sentiment is genuinely about \
someone/something else, with Curry only a reference point or aside. \
Example: "James 'Curry' Harden" -> incidental (Curry used only as a nickname \
for Harden -- no sentiment about Curry himself is being expressed).
- "comparative": the comment's core point is an explicit comparison where \
sentiment is genuinely SPLIT or the comparison itself (not either player \
individually) is the subject. Use this only when about_curry doesn't fit \
better -- i.e. when you truly cannot say the comment is substantively about \
Curry specifically. Example: "Harden's a ball hog, at least Curry knows how \
to play team ball" -> comparative (frustration is about Harden; Curry is a \
positive reference point, not really being evaluated himself).
- "unclear": no real sentiment expressed, or genuinely ambiguous.

TIE-BREAK RULE: when in doubt between "about_curry" and "comparative", prefer \
"about_curry" if the comment makes a specific, substantive claim about Curry \
(his play, health, stats, character) even if phrased via comparison. Only use \
"comparative" when the sentiment is truly about the comparison/matchup itself, \
or is clearly directed elsewhere.

If a comment is just a bare list of player names or has no real sentence \
structure, treat it as a single item and pick the ONE category/sentiment \
that best fits the comment as a whole -- do not analyze each name separately.

If a comment is a factual question with no real sentiment (e.g. "What is \
Curry's nickname?"), or is a joke/meme with no genuine sentiment, still \
respond with the required JSON format -- use "unclear" as the subject with \
"neutral" sentiment and score 0.0. Do NOT answer the question conversationally, \
and do NOT generate additional jokes, examples, or continuations -- only \
classify the single comment given.

INSTRUCTIONS:
First, in 1-2 sentences, briefly reason about who the sentiment is actually \
directed at. Then, on a new line, write exactly "FINAL_ANSWER:" followed by a \
JSON object with this exact format:
{"subject": "about_curry" | "incidental" | "comparative" | "unclear", \
"sentiment": "positive" | "negative" | "neutral", "score": <float from -1.0 to 1.0>}

For "score": use a precise, continuous value reflecting actual intensity -- \
avoid defaulting to round numbers like -0.5 or 0.8 unless the sentiment is \
genuinely that extreme. A mild criticism might be -0.2, a strong one -0.7, etc. \
If subject is "incidental", sentiment/score should reflect feeling toward \
Curry specifically (often close to 0 if none is really expressed toward him).
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


def extract_json(raw):
    if "FINAL_ANSWER:" in raw:
        after = raw.split("FINAL_ANSWER:", 1)[1].strip()
        match = re.search(r"\{.*?\}", after, re.DOTALL)
        if match:
            return json.loads(match.group(0))

    matches = re.findall(r"\{.*?\}", raw, re.DOTALL)
    if not matches:
        raise json.JSONDecodeError("no JSON object found", raw, 0)
    return json.loads(matches[-1])


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
                temperature=0.2,
                max_tokens=250,
            )
            raw = completion.choices[0].message.content.strip()

            if looks_like_refusal(raw):
                print(f"    model refused (detected pattern) -- not retrying: {raw[:100]!r}")
                return None, None, None

            parsed = extract_json(raw)
            return parsed.get("subject"), parsed.get("sentiment"), parsed.get("score")
        except json.JSONDecodeError as e:
            last_error = f"JSON parse failed on response: {raw!r} ({e})"
            delay = FORMAT_RETRY_DELAY_SECONDS
        except Exception as e:
            last_error = str(e)
            delay = API_ERROR_RETRY_DELAY_SECONDS

        if attempt < MAX_RETRIES:
            time.sleep(delay)

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