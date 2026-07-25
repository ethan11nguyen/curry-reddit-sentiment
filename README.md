# Curry Reddit Sentiment Analysis
Analyzes Reddit sentiment toward Stephen Curry during May 2015, covering his 2014-15 NBA MVP and championchip playoff run, and correlates that fan sentiment against his real game box scores. Built as a portfolio project to showcase skills in SQL, Python, Power BI, and statistical modeling.

## Approach & Pivots
**Original Plan:** analyze fan sentiment on Shai Gilgeous Alexander (SGA), live-scraped from r/nba from Reddit's API (PRAW), and finding the correlation of his fans sentiment versus his game statistics. However, Reddit's API requires an access request with no guaranteed approval time.

**Attempted workarounds:**
- Scraping Reddit's public .json for the same data on SGA, however was blocked by Reddit's anti-bot layer
- X/Twitter's API was considered, but found that recent search only covers the last 7 days on the affordable tier, and full-archive historical search requires a considerable amount of money.

**The Pivot:** Switched to a static historical dataset from Kaggle on all Reddit comments in May 2015, and moved from SGA to Stephen Curry, since May 2015 happens to contain Curry's MVP award announcement (May 4) and his championship run with the Warriors.

## Sentiment Scoring: VADER vs. LLM
Sentiment was initially scored using the VADER lexicon-based natural language processor. However after a manual review of 150 comments for validation, the VADER NLP was found to have two problems:

1. Subject misattribution: Many comments mention Curry in passing while the sentiment is about someone else, however VADER scores the entire comment as sentiment against/for Curry.
2. Slang: Sports trash-talk scores as strongly negative under the VADER lexicon. For example one comment saying "Shit on em curry" was given a negative sentiment score of -0.5574, but in reality, the sentiment was positive. 

To address this, a pivot to using an LLM (Llama 3.1 8B via Hugging Face) was taken and given a prompt to return both the sentiment and who its actually about (`about_curry` / `incidental` / `comparative` / `unclear`). This was validated against the same 150 comment sample, and found that the LLM's `about_curry` reached ~78% accuracy against manual labels. However `incidental` and `comparative` was still unreliable after prompt tuning. The final sentiment scoring model used in the statistical analysis settled on the LLM after prompt adjusting, `model_version = llm_statified_v2`, filtered to comments only `about_curry`. VADER's score (model_version = `vader_sentence_filtered_v1`) is kept as a secondary comparison, given its documented limitations above. 

## Known Limitations
- **Statistical power:** The dataset covers only 11 games in May 2015, which limits the statistical power of the OLS and event study models. A power analysis found that with only n=11 game days and 4 predictors (points, plus/minus, win/loss, home/away), giving a residual df=6, the minimum $R^2$ to detect effect would be roughly 69%. This sample size restriction should be read as "this study could not confirm a relationship" rather than "no relationship exists". 
- **LLM subjust-attribution reliability:** The LLM sentiment pipeline classifies each comment's subject as either `about_curry`, `incidental`, `comparative`, or `unclear`. A random sample of 150 comments were manually labeled, and the `about_curry` label reached 78.3% binary accuracy. However, `incidental` and `comparative` were not reliable even after prompt tuning, and were excluded from the statistical analysis. Of the 1540 stratified sample comments scored by the LLM, only 640 (~41%) were classified `about_curry`, and the remaining were excluded.
- **Sample coverage mismatch between VADER and the LLM:** The two sentiment scoring approaches were not run on the same population of comments. While the VADER nlp model scored the entire population of ~19k comments, the LLM sentiment analysis was performed on a stratified sample of ~50 comments a day (~1540 total) to accommodate API rate limits and processing times. As a result, VADER and LLM scores are not directly comparable measurements of the same comments.
- **Data source constraints:** The Reddit comment data comes from a static Kaggle dataset covering all Reddit comments for May 2015, and then filtered to r/nba and to comments regarding Stephen Curry. 
- **LLM mistakes:** The LLM labeled n=2 comments an off-schema sentiment labeled as "concerned" outside the expected positive/neutral/negative set. These were left in the data rather than dropped. The LLM pipeline also encountered 10 failures (of 1550 comments) regarding content-moderation refusals and JSON parsing failures during scoring, which were handled via a three-attempt retry loop (1s delay for JSON-format misses, 5s for real API errors), with content-moderation refusals detected and skipped immediately rather than retried. Comments still failing after that were logged and excluded, not inserted as null.

## Tech Stack
- Database: PostgreSQL 16, in Docker
- Data sources: Kaggle May 2015 Reddit Comments dataset, `nba_api`
- Sentiment scoring: VADER, Llama 3.1 8B (Hugging Face Inference Providers)
- Statistical Modeling: Python, statsmodels (OLS, robust SEs, ARMA), scipy
- Tools: Git, Docker, DBeaver

## Repo Structure

├── docker-compose.yml          # Postgres container definition
├── requirements.txt            # Python dependencies
├── .env.example                # Template for required environment variables
│
├── sql/
│   ├── 01_reddit_schema.sql               # Reddit comments/posts table schema
│   ├── 02_player_stats_schema.sql         # Curry game log table schema
│   ├── 03_aggregation_queries.sql         # Daily sentiment + game performance views (modeling input)
│   └── 04_validation_comparison_queries.sql  # LLM vs. VADER confusion matrix (comment-level)
│
├── scripts/
│   ├── load_kaggle_reddit_dataset.py      # Loads the Kaggle May 2015 Reddit comments dataset
│   ├── nba_stats_pull.py                  # Pulls Curry's 2014-15 game logs via nba_api
│   ├── sentiment_scoring.py               # VADER scoring
│   ├── llm_sentiment_scoring.py           # v1 LLM prompt, run against the 150-comment validation sample
│   ├── llm_sentiment_scoring_v2.py        # reverted/overcorrected prompt version
│   ├── llm_stratified_scoring.py          # final stratified LLM scoring run (50/day)
│   ├── export_validation_sample.py        # Exports the 150-comment manual validation sample
│   ├── reorder_validation_csv_columns.py  # Reorders LLM columns before VADER in validation CSVs
│   ├── statistical_modeling.py            # OLS, robust SEs, ARMA, event study, power analysis
│   └── test_hf_inference.py               # Hugging Face Inference Providers connectivity test
│
├── docs/
│   ├── validation_sample.csv              # Manually labeled 150-comment sample
│   ├── validation_sample_with_llm.csv     # + LLM labels (v1 prompt)
│   └── validation_sample_with_llm_v2.csv  # + LLM labels (v2 prompt, reverted)
│
├── powerbi/
│   └── curry_sentiment_powerbi.pbix       # Dashboard: sentiment timeline, win/loss, LLM/VADER validation
│
└── archive/
    └── sga_json_scraper.py                # Abandoned SGA live-scraping approach (see Approach & Pivots)

## Setup Instructions

**Prerequisites:** Docker, Python 3.x (Anaconda environment recommended), a Hugging Face account with an Inference Providers API token.

1. Clone the repo and `cd` into it.
2. Copy `.env.example` to `.env` and fill in your Postgres credentials and `HUGGINGFACE_API_TOKEN`.
3. Install Python dependencies: `pip install -r requirements.txt`
4. Start Postgres: `docker compose up -d`
5. Run the schema files against the database (via `psql -f` or DBeaver, in order):
   - `sql/01_reddit_schema.sql`
   - `sql/02_player_stats_schema.sql`
6. Load the data:
   - `python scripts/load_kaggle_reddit_dataset.py`
   - `python scripts/nba_stats_pull.py`
7. Run sentiment scoring:
   - `python scripts/sentiment_scoring.py` (VADER, full corpus)
   - `python scripts/llm_stratified_scoring.py` (LLM, production 50/day stratified sample)
   - *(Optional, for reproducing the validation writeup)*: `python scripts/llm_sentiment_scoring.py` and `llm_sentiment_scoring_v2.py` against the 150-comment sample in `docs/validation_sample.csv`
8. Run the aggregation views:
   - `sql/03_aggregation_queries.sql`
   - `sql/04_validation_comparison_queries.sql`
9. Run the statistical modeling script: `python scripts/statistical_modeling.py`
10. (Optional) Open `powerbi/curry_sentiment_powerbi.pbix` in Power BI Desktop — point the data source at your local Postgres instance.

