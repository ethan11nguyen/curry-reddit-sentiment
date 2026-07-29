"""
export_validation_sample.py

Pulls a random sample of comments and exports them to CSV for manual
labeling. The goal: empirically measure how often a comment's overall
sentiment is genuinely about SGA vs. incidental/comparative (e.g. SGA
mentioned only as a comparison point for another player).

IMPORTANT: this sample is drawn from `stratified_sample`, NOT the raw
`comments` table. stratified_sample is the ~28,500-comment population
that will actually get scored by VADER/LLM in production (see
stratified_sample_comments.py for how it was built: 50/day baseline
across all 273 days + 150/day boost on the 99 actual game days).
Drawing the validation set from this same table -- rather than the full
7.18M-comment corpus -- means your measured accuracy reflects the exact
population you're trusting for the real analysis, not a differently-
distributed sample. Since stratified_sample already contains the right
mix of baseline vs. game-day-boost rows, a plain random draw from it
naturally preserves that ~48%/52% proportion without needing to
stratify again here.

This is a measurement step, not part of the automated pipeline. You label
the sample by hand, then use the results to report a concrete noise-rate
figure in your findings writeup (e.g. "in a random sample of N comments,
X% were incidental/comparative mentions rather than direct SGA sentiment"),
rather than just noting the limitation vaguely.

Workflow:
    1. Run this script -> produces docs/validation_sample.csv
    2. Open the CSV (Excel, Numbers, Google Sheets, whatever) and fill in
       the `manual_subject_label` column for each row using the values:
         - "about_sga"    : sentiment is genuinely directed at SGA
         - "incidental"   : SGA mentioned in passing / as a reference
                             point, sentiment is about someone/something else
         - "comparative"  : sentence directly compares SGA to another
                             player -- sentiment is ambiguous/split
         - "unclear"      : can't tell / not really about sentiment at all
    3. Once labeled, come back and I'll help you compute agreement rates
       between VADER's / the LLM's scores and your manual labels.

NOTE: the LEFT JOIN to sentiment_scores will return NULLs for the vader_*
columns until sentiment_scoring.py has actually been run against this
sample -- that's expected if you're exporting the validation sample
before running VADER. Fine either way; the manual_subject_label column
is what matters for labeling and doesn't depend on VADER having run yet.

Run:
    python scripts/export_validation_sample.py
"""

import csv
import os
import random

import psycopg2
from dotenv import load_dotenv

SAMPLE_SIZE = 150
RANDOM_SEED = 42  # fixed seed so the sample is reproducible
OUTPUT_PATH = "docs/validation_sample.csv"

load_dotenv()

PG_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": os.getenv("POSTGRES_PORT", "5432"),
    "dbname": os.getenv("POSTGRES_DB", "sga_sentiment"),
    "user": os.getenv("POSTGRES_USER", "sga_admin"),
    "password": os.getenv("POSTGRES_PASSWORD"),
}


def get_pg_conn():
    return psycopg2.connect(**PG_CONFIG)


def fetch_sample(pg_conn, size, seed):
    # Draw from stratified_sample (the ~28,500-comment production
    # population), not the raw comments table -- see module docstring.
    # Use Postgres's own random sampling (TABLESAMPLE isn't seedable in a
    # simple way across versions, so use ORDER BY random() with setseed()
    # for reproducibility).
    with pg_conn.cursor() as cur:
        cur.execute("SELECT setseed(%s)", (seed / 1000.0,))  # setseed wants [-1, 1]
        cur.execute(
            """
            SELECT
                c.comment_id, c.body, ss.sample_tier,
                s.sentiment_score, s.sentiment_label
            FROM stratified_sample ss
            JOIN comments c ON c.comment_id = ss.comment_id
            LEFT JOIN sentiment_scores s
                ON c.comment_id = s.comment_id
                AND s.model_version = 'vader_sentence_filtered_v1'
            WHERE c.body IS NOT NULL
            ORDER BY random()
            LIMIT %s
            """,
            (size,),
        )
        return cur.fetchall()


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    pg_conn = get_pg_conn()
    try:
        rows = fetch_sample(pg_conn, SAMPLE_SIZE, RANDOM_SEED)
    finally:
        pg_conn.close()

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "comment_id",
                "body",
                "sample_tier",
                "vader_sentiment_score",
                "vader_sentiment_label",
                "manual_subject_label",  # blank -- you fill this in
                "notes",  # blank -- optional free text
            ]
        )
        for comment_id, body, sample_tier, score, label in rows:
            writer.writerow([comment_id, body, sample_tier, score, label, "", ""])

    print(f"Exported {len(rows)} comments to {OUTPUT_PATH}")
    print("Fill in the 'manual_subject_label' column, then bring it back for analysis.")


if __name__ == "__main__":
    main()