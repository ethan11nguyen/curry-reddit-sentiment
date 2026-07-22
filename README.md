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

## Tech Stack
- Database: PostgreSQL 16, in Docker
- Data sources: Kaggle May 2015 Reddit Comments dataset, `nba_api`
- Sentiment scoring: VADER, Llama 3.1 8B (Hugging Face Inference Providers)
- Statistical Modeling: Python, statsmodels (OLS, robust SEs, ARMA), scipy
- Tools: Git, Docker, DBeaver

## Repo Structue


## Setup Instructions


