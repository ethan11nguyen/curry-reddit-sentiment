# SGA Reddit Sentiment Analysis

Analyzes Reddit sentiment toward Shai Gilgeous-Alexander (SGA) across the
Oklahoma City Thunder's full 2024-25 season -- the season in which SGA
won regular season MVP, Finals MVP, and the championship -- and tests
whether that fan sentiment correlates with his real game performance.

Built as a portfolio project to showcase skills in data engineering,
SQL, Python, LLM-based text classification, and statistical modeling.

See [`docs/findings.md`](docs/findings.md) for the full results writeup.

## Data Pipeline

Reddit comment data comes from pushshift's r/nba archives (via Academic
Torrents), covering October 2024 through June 2025 -- 7.18 million
comments total. These are streamed and filtered for r/nba directly from
the compressed monthly archives without full disk decompression
(`scripts/filter_subreddit_streaming.py`), then loaded into Postgres.

A stratified sample of ~23,000 SGA-mentioning comments was drawn for
sentiment scoring: a daily baseline for time-series coverage, plus
additional sampling on actual game days, since those are the days the
performance-correlation analysis depends on. See
`scripts/stratified_sample_comments.py` for the full sampling design.

Game performance data comes from `nba_api`, covering SGA's full 2024-25
regular season and playoff game logs.

## Sentiment Scoring: VADER vs. LLM

Sentiment was scored two ways, run against the *same* ~23,000-comment
sample for a direct, apples-to-apples comparison:

1. **VADER** (lexicon-based): fast and free, but has no ability to tell
   *who* a comment's sentiment is actually about -- many comments
   mention SGA only in passing while the real sentiment is directed at
   another player, a comparison, or the team as a whole.
2. **LLM** (Llama 3.3 70B via Hugging Face Inference Providers): given
   the full comment text, classifies both subject (`about_sga` /
   `not_about_sga` / `unclear`) and sentiment.

The LLM prompt was validated against a 150-comment hand-labeled sample
and refined through several rounds of targeted fixes based on specific
observed errors (see `scripts/llm_stratified_scoring.py` for the full
prompt and its documented rationale). A smaller 8B model plateaued
around 67% accuracy despite prompt refinement -- diagnosed as a genuine
model capability limit rather than a fixable prompt issue. Upgrading to
70B with the same refined prompt reached **94.7% accuracy** against the
hand-labeled set, which is the model/prompt combination used for the
full production run.

## Key Finding

Same-day game performance does **not** predict same-day sentiment -- but
strongly predicts **next-day** sentiment. NBA games tip off in the
evening, so most genuine fan reaction lands on the following calendar
day rather than the game's own day; correcting for this timing turned a
null result (R-squared=0.04) into a strong, statistically significant
one (R-squared=0.26, p<0.0001). A second naive finding -- a sentiment
shift around the MVP announcement -- turned out to be confounded with
the concurrent start of the playoffs, and lost significance once
controlled for.

## Dashboard

![Same-day vs. next-day sentiment correlation, full-season timeline, and validation metrics](docs/images/page1_screenshot.png)

*Page 1: the same-day/next-day scatter comparison that surfaced the
headline finding, R² and game-count context, and a full-season sentiment
timeline marking the MVP announcement and playoff start.*

![LLM vs. VADER confusion matrix and event study comparison](docs/images/page2_screenshot.png)

*Page 2: validation accuracy, the LLM-vs-VADER confusion matrix, and the
naive-vs-playoff-controlled event study comparison showing the confound
catch. See `powerbi/sga_powerbi.pbix` for the full interactive
dashboard (Power BI Desktop required).*

Full methodology, all statistical tests, and complete results are in
[`docs/findings.md`](docs/findings.md).

## Known Limitations

- **Sampling scope**: sentiment scoring (both VADER and LLM) covers the
  ~23,000-comment stratified sample, not the full 7.18M-comment corpus --
  a deliberate tradeoff to keep both methods scoped identically for a
  clean comparison, at the cost of full-corpus coverage.
- **LLM subject-classification accuracy**: 94.7% on the validation
  sample, with remaining errors roughly evenly split between over- and
  under-attribution (not systematically biased one direction) --
  consistent with residual, largely irreducible ambiguity in genuinely
  hard cases rather than a fixable pattern.
- **Statistical power**: an earlier single-month pilot phase of this
  project (n=11 games) had so little power that even a large effect
  would likely have gone undetected -- any null result there was
  uninterpretable. The full-season analysis (n=99 games) has 87-99.9%
  power to detect medium/large effects, making its null and positive
  results both far more trustworthy.
- **Correlational, not causal**: all relationships reported are
  correlational. The next-day timing result is a robust correlation; it
  doesn't independently establish that performance *causes* the
  sentiment shift, though it's the substantively plausible direction.
- **Single-season, single-player**: findings are specific to SGA's
  2024-25 season and shouldn't be assumed to generalize elsewhere
  without independent testing.

## Tech Stack
- **Database**: PostgreSQL 16, in Docker
- **Data sources**: pushshift Reddit archives (via Academic Torrents),
  `nba_api`
- **Sentiment scoring**: VADER, Llama 3.3 70B (Hugging Face Inference
  Providers)
- **Statistical modeling**: Python, statsmodels (OLS, robust SEs
  including HAC/Newey-West, ARMA, event study, power analysis via
  noncentral F, VIF), scipy
- **Tools**: Git, Docker, DBeaver

## Repo Structure

```
├── docker-compose.yml                        # Postgres container definition
├── requirements.txt                          # Python dependencies
├── .env.example                              # Template for required environment variables
│
├── sql/
│   ├── 01_reddit_schema.sql                  # Comments table schema (typed columns + JSONB)
│   ├── 02_player_stats_schema.sql            # SGA game log table schema
│   ├── 03_aggregation_queries.sql            # Daily sentiment + game performance views (modeling input)
│   └── 04_validation_comparison_queries.sql  # LLM vs. VADER confusion matrix (comment-level)
│
├── scripts/
│   ├── filter_subreddit_streaming.py         # Streams pushshift archives, filters to r/nba
│   ├── load_pushshift_comments.py            # Loads filtered comments into Postgres
│   ├── nba_stats_pull.py                     # Pulls SGA's 2024-25 game logs via nba_api
│   ├── stratified_sample_comments.py         # Builds the ~23K-comment stratified sample
│   ├── vader_sentiment_scoring.py            # VADER scoring, stratified sample
│   ├── llm_sentiment_scoring.py              # LLM prompt validation, run against the 150-comment sample
│   ├── llm_stratified_scoring.py             # Full-scale LLM scoring run, stratified sample
│   ├── export_validation_sample.py           # Exports the 150-comment manual validation sample
│   ├── check_llm_accuracy.py                 # Compares LLM output against manual labels
│   ├── test_hf_inference.py                  # Hugging Face Inference Providers connectivity test
│   ├── statistical_modeling.py               # OLS, robust SEs, ARMA, event study, power analysis
│   ├── newey_west_comparison.py              # HAC/Newey-West robust SEs, checking autocorrelation-corrected significance
│   └── multicollinearity_check.py            # VIF + reduced-model comparison for win_loss/plus_minus/points
│
├── docs/
│   ├── findings.md                           # Full results writeup
│   ├── validation_sample.csv                 # Manually labeled 150-comment sample
│   └── validation_sample_with_llm.csv        # + LLM labels
│
└── archive/
    └── may2015_kaggle_pipeline/              # Earlier project phase -- see docs/findings.md for context
```

## Setup Instructions

**Prerequisites:** Docker, Python 3.x (Anaconda environment recommended),
a Hugging Face account with an Inference Providers API token and billing
configured (production-scale LLM scoring costs roughly $50-60 depending
on provider routing -- see cost notes in `llm_stratified_scoring.py`).

1. Clone the repo and `cd` into it.
2. Copy `.env.example` to `.env` and fill in your Postgres credentials
   and `HF_TOKEN`.
3. Install Python dependencies: `pip install -r requirements.txt`
4. Start Postgres: `docker compose up -d` (schema auto-initializes from
   `sql/` on first run)
5. Acquire the raw data:
   - Download pushshift r/nba monthly comment archives (Academic
     Torrents) for your target date range
   - Filter each month: `python scripts/filter_subreddit_streaming.py <input.zst> <output.jsonl>`
   - Combine the monthly outputs into one file (e.g.
     `data/raw/nba_full_season.jsonl`)
6. Load the data:
   - `python scripts/load_pushshift_comments.py data/raw/nba_full_season.jsonl`
   - `python scripts/nba_stats_pull.py`
7. Build the sampling and scoring pipeline:
   - `python scripts/stratified_sample_comments.py`
   - `python scripts/vader_sentiment_scoring.py`
   - `python scripts/llm_stratified_scoring.py` (production LLM run -- see cost note above)
   - *(Optional, for reproducing the validation writeup)*:
     `python scripts/export_validation_sample.py`, hand-label
     `manual_subject_label`, then `python scripts/llm_sentiment_scoring.py`
     and `python scripts/check_llm_accuracy.py`
8. Run the aggregation views:
   - `psql -f sql/03_aggregation_queries.sql`
   - `psql -f sql/04_validation_comparison_queries.sql`
9. Run the statistical modeling script: `python scripts/statistical_modeling.py`