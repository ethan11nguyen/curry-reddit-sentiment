"""
llm_stratified_scoring.py

Scores the FULL stratified_sample (~23,093 comments) using the same
Llama-3.3-70B-Instruct model and prompt validated against the 150-comment
manual validation set (94.7% subject-classification accuracy -- see
llm_sentiment_scoring.py and docs/validation_sample_with_llm.csv for the
validation process this prompt was refined against).

Writes results into `sentiment_scores` with model_version='llm_stratified_v1',
alongside the existing VADER scores (model_version='vader_sentence_filtered_v1')
already in that table for the same stratified_sample population -- both
methods now cover the exact same ~23K comments, enabling a clean
VADER-vs-LLM-vs-manual-label comparison (see sql/04_validation_comparison_queries.sql,
which will need rewriting to join against this new model_version once this
finishes running).

COST/TIME NOTE: at ~23,093 comments and REQUEST_DELAY_SECONDS below, this
is a multi-hour run, not a quick script. Estimate before starting:
23,093 * REQUEST_DELAY_SECONDS gives a rough floor in seconds (add real
API response time on top -- 70B models respond slower than 8B). Worth
running this unattended (e.g. overnight), same as the earlier pushshift
filtering step.

RESUME SUPPORT: unlike the CSV-based validation script, this checks
already-scored comment_ids directly in Postgres (via a query at startup),
not a local file -- safe to stop and restart without losing progress or
re-spending API calls on already-scored rows.

Setup:
    pip install huggingface_hub psycopg2-binary python-dotenv --break-system-packages
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
MODEL = "meta-llama/Llama-3.3-70B-Instruct"
PROVIDER = "auto"

REQUEST_DELAY_SECONDS = 1.0
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 5
DB_WRITE_BATCH_SIZE = 50  # commit to Postgres every N successful scores

# Identical to llm_sentiment_scoring.py's validated prompt -- see that
# file's module docstring for the full rationale behind each rule
# (3-way schema, tie-break, short-comment default, multi-player-mention
# test, pronoun resolution, missing-context handling). Kept byte-for-byte
# in sync deliberately since this prompt was specifically validated
# against the 150-comment manual sample before being scaled up here.
SYSTEM_PROMPT = """You analyze Reddit comments from r/nba for sentiment specifically about \
the basketball player Shai Gilgeous-Alexander (SGA), point guard for the \
Oklahoma City Thunder, 2024-25 regular season MVP and Finals MVP. Many \
comments mention SGA only in passing while actually expressing sentiment \
about someone or something else -- comparing him to another player \
(Luka Doncic, Nikola Jokic, Joel Embiid, etc.), using him as a reference \
point, or discussing the Thunder as a team rather than SGA specifically. \
Your job is to identify whether the comment expresses genuine sentiment \
TOWARD SGA himself, and if so, what that sentiment is.

Respond with ONLY a JSON object, no other text, in this exact format:
{"subject": "about_sga" | "not_about_sga" | "unclear", \
"sentiment": "positive" | "negative" | "neutral", "score": <float from -1.0 to 1.0>}

Field meanings:
- "subject": "about_sga" if the sentiment is genuinely directed at SGA \
himself, INCLUDING comments that compare him to another player as long as \
the substantive point is about SGA (e.g. "SGA gets to the line more than \
Luka ever did" -- about_sga, since the claim is about SGA's foul-drawing). \
Use "not_about_sga" if SGA is mentioned but the real sentiment is about \
someone/something else -- another player, the Thunder as a team, the \
broadcast, refs, etc. -- with SGA only as an aside or reference point. \
Use "unclear" only if you genuinely cannot tell, or there is no real \
sentiment expressed at all (e.g. a pure factual statement with no opinion).
- TIE-BREAK RULE: if a comment could reasonably be read as either \
about_sga or not_about_sga, prefer "about_sga" as long as SGA is the \
grammatical subject of the sentiment-bearing clause, even if another \
player is named nearby for context or comparison.
- "sentiment": the overall sentiment label. If subject is "not_about_sga", \
this should reflect sentiment toward SGA specifically (often "neutral" if \
none is actually expressed toward him).
- "score": -1.0 (very negative) to 1.0 (very positive), 0.0 = neutral. \
This should reflect sentiment toward SGA specifically, not the whole comment.

IMPORTANT -- DEFAULT TOWARD about_sga ON SHORT/TERSE COMMENTS: Reddit \
comments are often a single short clause or fragment, not a full \
elaborated argument. Do NOT require substantial elaboration before \
calling something about_sga. A short, direct evaluative statement that \
names SGA is about_sga even if it's just a few words -- e.g. "Sga is a \
scrub", "Shai is playing 4 quarters.", "I mean Shai is young lol", and \
even a bare one-word reply like "SGA" (e.g. answering "who's your MVP \
pick") are all about_sga. Terseness is not evidence of \
ambiguity -- judge the CONTENT of what's said about SGA, however brief, \
not the length of the comment.

MULTI-PLAYER MENTIONS -- test whether the claim is ABOUT the named \
individuals or just uses them as EXAMPLES of a bigger category: \
if SGA is named alongside other players as an equal, direct subject of \
one specific claim (about their play, stats, decisions, or performance), \
it's about_sga -- this holds regardless of how many players are named \
together (2, 3, 5+). E.g. "SGA and Jokic decided they don't want the \
MVP no more" (2 players, both are the direct subject) and "sga jdub chet \
ihart caruso going to put us into fifth apron" (5 players, all equal \
subjects of one payroll claim) are BOTH about_sga. But if SGA is named \
only as one illustrative example within a list supporting some broader \
point that isn't really about any of the individuals -- e.g. "I'd say \
the % of superstars have gotten more diverse. Like Jokic, Wemby, \
Giannis, and yes SGA (he's not American)" -- that's not_about_sga, since \
the claim is about a league-wide trend, not about SGA's own play or \
performance specifically.

PRONOUN RESOLUTION -- only default to "unclear" if the pronoun's \
antecedent is COMPLETELY ABSENT from the comment. If SGA/Shai is named \
ANYWHERE in the same comment -- before OR after the pronoun -- resolve \
the pronoun to him and classify normally; do not treat a pronoun as \
inherently ambiguous just because it appears before the name. E.g. \
"Shai.. he's been robbed before" is about_sga (name appears immediately \
before "he", trivially resolved). Only use "unclear" for pronouns like \
"He's been a pleasant surprise. His lateral quickness still leaves much \
to be desired" where NO name appears anywhere in the comment at all -- \
there is genuinely no way to know who "he" refers to.

MISSING EXTERNAL CONTEXT -- separately from pronouns, some comments \
reference something you cannot see (a stat, a game, a prior comment) \
using vague deictic words like "this" or "that" with no definition \
anywhere in the comment itself. E.g. "Idk man giannis, SGA, and jokic \
bad games are never this bad" -- "this bad" refers to some specific \
performance never described in the comment, so the actual claim being \
made is unknowable. Use "unclear" for these cases too, distinct from \
(but similarly to) unresolved pronouns.

Examples:
- "SGA is getting to the line 12 times a game, refs are basically giving \
him free points" -> {"subject": "about_sga", "sentiment": "negative", \
"score": -0.4} (criticism of SGA's playstyle/foul-drawing)
- "Shai time. Bucket after bucket in the 4th, nobody can guard him" -> \
{"subject": "about_sga", "sentiment": "positive", "score": 0.9}
- "Jokic is a way better passer, SGA just scores in isolation" -> \
{"subject": "not_about_sga", "sentiment": "neutral", "score": 0.0} \
(the substantive claim/criticism is about Jokic's superior passing, \
SGA is only the comparison point)
- "Thunder defense has been elite all year, SGA included" -> \
{"subject": "not_about_sga", "sentiment": "neutral", "score": 0.0} \
(praise is directed at the team, SGA mentioned only incidentally)
- "SGA > Luka, not even close this year" -> {"subject": "about_sga", \
"sentiment": "positive", "score": 0.7} (comparison, but the substantive \
claim is a positive assertion about SGA specifically, per the tie-break rule)
- "Sga is a scrub" -> {"subject": "about_sga", "sentiment": "negative", \
"score": -0.8} (short and blunt, but a direct evaluative claim about SGA \
-- don't require more elaboration than this)
- "SGA" -> {"subject": "about_sga", "sentiment": "neutral", "score": 0.0} \
(a bare name in reply to some other question, e.g. an MVP pick -- still \
about_sga even with zero elaboration; score neutral since no sentiment \
words are present)
- "He's been a pleasant surprise. His lateral quickness still leaves \
much to be desired" -> {"subject": "unclear", "sentiment": "neutral", \
"score": 0.0} (no name anywhere in this comment -- "He" is never tied \
to SGA, could be someone else discussed earlier in the thread that you \
can't see)
- "Shai.. he's been robbed before" -> {"subject": "about_sga", \
"sentiment": "positive", "score": 0.3} (name appears immediately before \
the pronoun -- trivially resolved, NOT an ambiguous case)
- "No way you just made this in response to the SGA highlight post \
lmaooo" -> {"subject": "not_about_sga", "sentiment": "neutral", \
"score": 0.0} (commentary about the post/other user's reaction, not \
sentiment toward SGA's play itself)
- "sga jdub chet ihart caruso going to put us into fifth apron" -> \
{"subject": "about_sga", "sentiment": "neutral", "score": 0.0} (5 \
players named as equal co-subjects of one payroll claim -- SGA is \
directly implicated, not just an example in a list)
- "I'd say the % of superstars have gotten more diverse. Like Jokic, \
Wemby, Giannis, and yes SGA (he's not American)" -> {"subject": \
"not_about_sga", "sentiment": "neutral", "score": 0.0} (SGA is one \
illustrative example supporting a claim about league-wide demographic \
trends, not a claim about his own play)
- "Idk man giannis, SGA, and jokic bad games are never this bad" -> \
{"subject": "unclear", "sentiment": "neutral", "score": 0.0} ("this bad" \
references some specific performance never described in the comment -- \
the actual claim is unknowable without context you don't have)
- "I need to see us win by more than 51 just so we can outmeat Shai yet \
again" -> {"subject": "not_about_sga", "sentiment": "neutral", \
"score": 0.0} (Shai is the numeric benchmark being chased, "us" -- not \
Shai -- is the actual subject of the claim)

Respond with the JSON object ONLY. No explanation, no examples, no extra text \
before or after it. Just the single JSON object.
"""

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise SystemExit("ERROR: HF_TOKEN not found in .env.")

PG_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": os.getenv("POSTGRES_PORT", "5432"),
    "dbname": os.getenv("POSTGRES_DB", "sga_sentiment"),
    "user": os.getenv("POSTGRES_USER", "sga_admin"),
    "password": os.getenv("POSTGRES_PASSWORD"),
}


def get_pg_conn():
    return psycopg2.connect(**PG_CONFIG)


def get_hf_client():
    return InferenceClient(api_key=HF_TOKEN, provider=PROVIDER)


def fetch_unscored_comments(conn):
    """Comments in stratified_sample that don't yet have an
    llm_stratified_v1 row in sentiment_scores. Ordering by comment_id
    just for a stable, resumable iteration order."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT c.comment_id, c.body
            FROM stratified_sample ss
            JOIN comments c ON c.comment_id = ss.comment_id
            WHERE NOT EXISTS (
                SELECT 1 FROM sentiment_scores s
                WHERE s.comment_id = c.comment_id
                AND s.model_version = %s
            )
            ORDER BY c.comment_id
            """,
            (MODEL_VERSION,),
        )
        return cur.fetchall()


def score_comment(client, body):
    """Returns (subject, sentiment, score) or (None, None, None) on failure."""
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

            return (
                parsed.get("subject"),
                parsed.get("sentiment"),
                parsed.get("score"),
            )
        except json.JSONDecodeError as e:
            last_error = f"JSON parse failed on response: {raw!r} ({e})"
        except Exception as e:
            last_error = str(e)

        if attempt < MAX_RETRIES:
            print(f"    attempt {attempt} failed ({last_error}), retrying...")
            time.sleep(RETRY_DELAY_SECONDS)

    print(f"    giving up after {MAX_RETRIES} attempts: {last_error}")
    return None, None, None


def write_batch(conn, rows):
    if not rows:
        return
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(
            cur,
            """
            INSERT INTO sentiment_scores
                (comment_id, model_version, sentiment_score, sentiment_label, subject_label, scored_at)
            VALUES %s
            ON CONFLICT (comment_id, model_version)
            DO UPDATE SET
                sentiment_score = EXCLUDED.sentiment_score,
                sentiment_label = EXCLUDED.sentiment_label,
                subject_label = EXCLUDED.subject_label,
                scored_at = EXCLUDED.scored_at
            """,
            rows,
        )
    conn.commit()


def main():
    print(f"Model: {MODEL} (provider={PROVIDER}), model_version={MODEL_VERSION}")

    read_conn = get_pg_conn()
    write_conn = get_pg_conn()

    todo = fetch_unscored_comments(read_conn)
    print(f"{len(todo):,} comments left to score "
          f"(already-scored rows are skipped automatically via DB check).\n")

    est_seconds = len(todo) * REQUEST_DELAY_SECONDS
    print(f"Estimated minimum time: {est_seconds/3600:.1f} hours "
          f"(real API response time adds more on top -- this is a floor, not a forecast).\n")

    client = get_hf_client()

    scored = 0
    failed = 0
    batch = []

    for i, (comment_id, body) in enumerate(todo, start=1):
        subject, sentiment, score = score_comment(client, body or "")
        scored_at = datetime.now(timezone.utc)

        if subject is None:
            failed += 1
        else:
            scored += 1
            batch.append((comment_id, MODEL_VERSION, score, sentiment, subject, scored_at))

        if len(batch) >= DB_WRITE_BATCH_SIZE:
            write_batch(write_conn, batch)
            batch = []

        if i % 100 == 0 or i == len(todo):
            print(f"  [{i:,}/{len(todo):,}] scored={scored:,} failed={failed:,}")

        time.sleep(REQUEST_DELAY_SECONDS)

    write_batch(write_conn, batch)  # flush any remaining partial batch

    read_conn.close()
    write_conn.close()

    print(f"\nDone. Scored {scored:,} new, failed {failed:,}. "
          f"Results written to sentiment_scores with model_version='{MODEL_VERSION}'.")


if __name__ == "__main__":
    main()