"""
llm_sentiment_scoring.py

Scores SGA-related sentiment using an LLM via Hugging Face's Inference
Providers (the current replacement for the old Serverless Inference API),
rather than a lexicon-based approach like VADER.

WHY: VADER (see sentiment_scoring.py) has two known weaknesses observed in
this dataset:
  1. Subject attribution -- it can't tell whether sentiment in a comment is
     actually about SGA vs. someone else SGA is being compared to.
  2. Domain slang -- phrases like "shit on 'em" are positive in sports
     trash-talk but score strongly negative in VADER's general-English
     lexicon.

An LLM, prompted correctly, can potentially handle both by reading the
FULL comment (not a sentence-filtered fragment) and reasoning about who the
sentiment is actually directed at and what the words mean in context.

SCHEMA CHANGE from the earlier 4-way subject classification
(about_sga / incidental / comparative / unclear): collapsed to 3 categories
here. The prior project found the LLM had real, consistent difficulty
distinguishing "incidental" from "comparative" specifically (this is the
SAME failure mode observed in the original Curry-era pipeline, where a v2
prompt attempting finer subcategory distinctions measurably REDUCED
accuracy and had to be reverted). Since nothing downstream actually
splits those two categories apart for separate analysis -- both get
excluded identically wherever subject_label != 'about_sga' is filtered --
asking the model to make that distinction was pure accuracy cost with no
analytical benefit. Collapsing them into a single "not_about_sga" bucket
turns this into a cleaner binary-plus-fallback classification problem.

The manual validation labels (docs/validation_sample.csv) can still use
the original finer categories if you want a documentable "X% of
not-about-SGA comments were comparative vs. incidental" footnote -- that
distinction just isn't asked of the LLM anymore, since human judgment
doesn't share the model's confusion here.

TIE-BREAK RULE: when genuinely ambiguous between about_sga and
not_about_sga, the prompt instructs the model to favor about_sga. This
mirrors what measurably improved binary about_[player] accuracy in the
prior project -- worth keeping as a deliberate, documented choice, not
an accident.

This script deliberately targets the SAME 150-comment validation sample
you're manually labeling (docs/validation_sample.csv), NOT the full
stratified_sample set -- run this first, compare all three (manual /
VADER / LLM), and decide whether it's worth scaling up to the full
~28,500-comment stratified_sample before spending the time/cost there.

Setup:
    pip install huggingface_hub --break-system-packages
    Add to .env:  HF_TOKEN=hf_your_token_here
    (create a token at https://huggingface.co/settings/tokens with
    "Make calls to Inference Providers" permission)

Run:
    python scripts/llm_sentiment_scoring.py
"""

import csv
import json
import os
import re
import time

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

INPUT_PATH = "docs/validation_sample.csv"
OUTPUT_PATH = "docs/validation_sample_with_llm.csv"

# Configurable -- swap this if the model is slow, unavailable, or you want
# to compare a second model. "auto" lets HF pick the fastest available
# backend provider for this model.
#
# MODEL CHOICE: upgraded from Llama-3.1-8B-Instruct to Llama-3.3-70B-Instruct
# after the 8B model plateaued around 66.7% subject-classification accuracy
# despite multiple rounds of prompt refinement (explicit tie-break rules,
# short-comment examples, pronoun-ambiguity cautions). The residual errors
# looked like a genuine reasoning-capability limit -- e.g. the model kept
# missing direct, unambiguous statements ("SGA and Jokic decided they don't
# want the MVP no more") that a human calls instantly, and over-attributing
# subject to SGA whenever he appeared in a list of 3+ named players. A
# larger model should handle this kind of "who is the real grammatical/
# semantic subject" reasoning more reliably. Cost remains low at this
# comment volume even at 70B scale.
MODEL = "meta-llama/Llama-3.3-70B-Instruct"
PROVIDER = "auto"

# Be gentle on rate limits -- free-tier Inference Providers access can be
# throttled. Adjust down if you're not hitting limits, up if you are.
REQUEST_DELAY_SECONDS = 1.5
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 5

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
    raise SystemExit("ERROR: HF_TOKEN not found in .env. See script docstring for setup.")


def get_client():
    return InferenceClient(api_key=HF_TOKEN, provider=PROVIDER)


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
                temperature=0.1,  # low temp for consistency, not creativity
                max_tokens=150,
            )
            raw = completion.choices[0].message.content.strip()

            # Models sometimes wrap JSON in markdown fences, or ignore
            # instructions and add explanatory text around it, despite the
            # prompt explicitly forbidding this. Try a straight parse first;
            # if that fails, fall back to extracting the first {...} block.
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


def main():
    print(f"Model: {MODEL} (provider={PROVIDER})")
    print(f"Reading from: {INPUT_PATH}\n")

    with open(INPUT_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames

    # Resume support: if OUTPUT_PATH already exists from a prior partial run,
    # load it and skip re-scoring rows that already succeeded (non-empty
    # llm_subject). This avoids burning credit re-doing successful rows
    # when a run gets interrupted partway (e.g. by a billing cap).
    already_scored = {}
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, newline="", encoding="utf-8") as f:
            prev_reader = csv.DictReader(f)
            for prev_row in prev_reader:
                if prev_row.get("llm_subject", "").strip():
                    already_scored[prev_row["comment_id"]] = prev_row
        print(f"Found existing output with {len(already_scored)} already-scored rows -- will skip those.\n")

    client = get_client()

    new_fields = ["llm_subject", "llm_sentiment", "llm_score"]
    out_fieldnames = fieldnames + [fld for fld in new_fields if fld not in fieldnames]

    scored = 0
    failed = 0
    skipped = 0

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


if __name__ == "__main__":
    main()