"""
llm_sentiment_scoring_v2.py

Second iteration of LLM-based sentiment scoring, after the v1 prompt scored
only 34.7% subject-classification accuracy against manual labels on the
150-row validation sample. Changes from v1:

1. Few-shot examples targeting the SPECIFIC failure modes observed in v1
   (e.g. v1 called "James 'Curry' Harden" incidental when it's a name-pun,
   and called a clear on-court judgment about Curry "incidental").
2. An explicit tie-break rule for the about_curry/comparative boundary,
   since several v1 "errors" were actually genuine category overlap, not
   model failure (e.g. a comment can use a comparison as rhetorical framing
   while still being substantively about Curry).
3. A brief reasoning step before the final JSON answer, rather than
   forcing an immediate category guess -- intended to reduce pattern-
   matching to a default label.
4. An explicit instruction against score clustering (v1 outputs were
   suspiciously bucketed at -0.8/-0.5/0.5/0.6 rather than continuous).

This script re-scores the SAME 150-row validation sample (docs/validation_sample.csv)
under a new model_version-style tag ('llm_v2'), writes to a separate output
file so v1 results aren't lost, and automatically prints accuracy stats
against your manual labels at the end -- no separate comparison step needed
to check whether the prompt changes actually helped.

Setup: same as llm_sentiment_scoring.py (HF_TOKEN in .env)
Run:
    python scripts/llm_sentiment_scoring_v2.py
"""

import csv
import json
import os
import re
import time
from collections import defaultdict

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

INPUT_PATH = "docs/validation_sample.csv"
OUTPUT_PATH = "docs/validation_sample_with_llm_v2.csv"

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
PROVIDER = "auto"

REQUEST_DELAY_SECONDS = 1.5
MAX_RETRIES = 3
API_ERROR_RETRY_DELAY_SECONDS = 5  # for real network/rate-limit errors
FORMAT_RETRY_DELAY_SECONDS = 1  # for JSON-format misses -- not rate-limit related, no need for a long wait

# Phrases indicating the model refused to engage (e.g. content moderation on
# the backend provider) -- retrying gains nothing here, since the refusal
# will almost certainly repeat. Detected to skip wasted retry attempts.
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

INSTRUCTIONS:
If a comment is a factual question with no real sentiment (e.g. "What is \
Curry's nickname?"), or is a joke/meme with no genuine sentiment, still \
respond with the required JSON format -- use "unclear" as the subject with \
"neutral" sentiment and score 0.0. Do NOT answer the question conversationally, \
and do NOT generate additional jokes, examples, or continuations -- only \
classify the single comment given, using the JSON format below.

If a comment is just a bare list of player names with no sentence structure, \
treat it as a single item and pick the ONE category/sentiment that fits best \
-- do not analyze each name separately.

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

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise SystemExit("ERROR: HF_TOKEN not found in .env.")


def get_client():
    return InferenceClient(api_key=HF_TOKEN, provider=PROVIDER)


def extract_json(raw):
    """
    Looks for the FINAL_ANSWER: marker first (expected format), extracting
    only the JSON object immediately following it via regex -- NOT parsing
    all trailing text, since the model occasionally goes off-script and
    produces multiple answer blocks (e.g. if a comment mentions several
    player names, it may try to analyze each one separately). Falls back
    to the last {...} block in the text if the marker is missing entirely.
    """
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
                max_tokens=250,  # more room for the reasoning step
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


def print_accuracy_report(rows):
    correct = sum(1 for r in rows if r["llm_subject"].strip() == r["manual_subject_label"].strip())
    scored_rows = [r for r in rows if r["llm_subject"].strip()]
    n = len(scored_rows)
    if n == 0:
        print("No scored rows to report on.")
        return

    print(f"\n=== v2 Accuracy Report ===")
    print(f"{correct}/{n} = {correct/n:.1%} exact match with manual labels")

    categories = ["about_curry", "incidental", "comparative", "unclear"]
    by_manual = defaultdict(list)
    for r in scored_rows:
        by_manual[r["manual_subject_label"].strip()].append(r)

    print("\nPer-category recall:")
    for cat in categories:
        group = by_manual.get(cat, [])
        if not group:
            continue
        match = sum(1 for r in group if r["llm_subject"].strip() == cat)
        print(f"  {cat}: {match}/{len(group)} = {match/len(group):.1%}")


def main():
    print(f"Model: {MODEL} (provider={PROVIDER})")
    print(f"Reading from: {INPUT_PATH}\n")

    with open(INPUT_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames

    already_scored = {}
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, newline="", encoding="utf-8") as f:
            for prev_row in csv.DictReader(f):
                if prev_row.get("llm_subject", "").strip():
                    already_scored[prev_row["comment_id"]] = prev_row
        print(f"Found existing v2 output with {len(already_scored)} already-scored rows -- will skip those.\n")

    client = get_client()
    new_fields = ["llm_subject", "llm_sentiment", "llm_score"]
    out_fieldnames = fieldnames + [fld for fld in new_fields if fld not in fieldnames]

    scored, failed, skipped = 0, 0, 0

    for i, row in enumerate(rows, start=1):
        comment_id = row.get("comment_id")

        if comment_id in already_scored:
            prev = already_scored[comment_id]
            row["llm_subject"] = prev["llm_subject"]
            row["llm_sentiment"] = prev["llm_sentiment"]
            row["llm_score"] = prev["llm_score"]
            skipped += 1
            continue

        body = row.get("body", "")
        print(f"[{i}/{len(rows)}] {comment_id}: scoring...")
        subject, sentiment, score = score_comment(client, body)

        row["llm_subject"] = subject if subject is not None else ""
        row["llm_sentiment"] = sentiment if sentiment is not None else ""
        row["llm_score"] = score if score is not None else ""

        if subject is None:
            failed += 1
        else:
            scored += 1

        time.sleep(REQUEST_DELAY_SECONDS)

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone. Scored {scored} new, skipped {skipped} already-done, failed {failed}. Output: {OUTPUT_PATH}")
    print_accuracy_report(rows)


if __name__ == "__main__":
    main()