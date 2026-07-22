"""
statistical_modeling.py

Statistical analysis of Curry sentiment (May 2015) vs. game performance and
the May 4 MVP announcement. Pulls from the daily_sentiment_and_performance
view (see sql/03_aggregation_queries.sql) and runs:

1. OLS: does game performance predict same-day sentiment?
2. Breusch-Pagan / White heteroskedasticity tests on the OLS residuals
3. Refit with HC1/HC3 robust standard errors
4. ARMA time series model on the daily sentiment series (with automatic
   fallback to AR(1) if ARMA(1,1) shows signs of an unreliable fit)
5. Event study: sentiment before vs. after the May 4 MVP announcement

IMPORTANT CAVEAT, stated upfront rather than buried in a footnote: only
~11 Curry games fall inside the May 2015 comment window this dataset
covers. That is a small sample for any game-level regression -- treat the
OLS coefficients and significance tests here as suggestive, not
definitive, and say so explicitly in any writeup. This script reports
exact sample sizes at each step so that limitation stays visible rather
than getting lost in the summary tables.

Setup:
    pip install pandas statsmodels psycopg2-binary python-dotenv scipy sqlalchemy --break-system-packages
Run:
    python scripts/statistical_modeling.py
"""

import os
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from dotenv import load_dotenv
from scipy import stats as scipy_stats
from sqlalchemy import create_engine
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.tsa.arima.model import ARIMA

load_dotenv()

PG_USER = os.getenv("POSTGRES_USER", "curry_admin")
PG_PASSWORD = os.getenv("POSTGRES_PASSWORD")
PG_HOST = os.getenv("POSTGRES_HOST", "localhost")
PG_PORT = os.getenv("POSTGRES_PORT", "5432")
PG_DB = os.getenv("POSTGRES_DB", "curry_sentiment")

# Which sentiment series to use as the primary outcome variable.
# 'llm_avg_score' is the validated, about_curry-filtered series (recommended
# primary). 'vader_avg_score' is kept available for a robustness comparison.
PRIMARY_SENTIMENT_COL = "llm_avg_score"


def load_data():
    # SQLAlchemy engine instead of a raw psycopg2 connection -- pandas
    # warns (harmlessly) on raw DBAPI2 connections; this avoids that noise.
    engine = create_engine(
        f"postgresql+psycopg2://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DB}"
    )
    df = pd.read_sql(
        "SELECT * FROM daily_sentiment_and_performance ORDER BY comment_date",
        engine,
    )
    df["comment_date"] = pd.to_datetime(df["comment_date"])
    return df


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ----------------------------------------------------------------------
# 1. OLS: performance -> same-day sentiment, on game days only
# ----------------------------------------------------------------------
def run_ols(df):
    section("1. OLS: game performance -> same-day sentiment")

    game_days = df[df["is_game_day"]].copy()
    n = len(game_days)
    print(f"Game days available for regression: n={n}")
    print("NOTE: this is a small sample. Coefficient estimates and p-values")
    print("should be treated as suggestive, not confirmatory, given n. A")
    print("single unusual game can meaningfully shift these results.\n")

    if n < 5:
        print("Too few game days to run a meaningful regression. Skipping.")
        return None, game_days

    formula = f"{PRIMARY_SENTIMENT_COL} ~ points + plus_minus + C(win_loss) + C(home_away)"
    model = smf.ols(formula=formula, data=game_days).fit()
    print(model.summary())
    return model, game_days


# ----------------------------------------------------------------------
# 2. Heteroskedasticity testing
# ----------------------------------------------------------------------
def test_heteroskedasticity(model, game_days):
    section("2. Heteroskedasticity tests (Breusch-Pagan, White)")

    if model is None:
        print("No model to test (skipped due to insufficient data).")
        return None

    exog = model.model.exog
    resid = model.resid

    bp_stat, bp_pvalue, _, _ = het_breuschpagan(resid, exog)
    print(f"Breusch-Pagan: LM stat={bp_stat:.4f}, p-value={bp_pvalue:.4f}")

    try:
        white_stat, white_pvalue, _, _ = het_white(resid, exog)
        print(f"White test:    stat={white_stat:.4f}, p-value={white_pvalue:.4f}")
    except Exception as e:
        white_pvalue = None
        print(f"White test could not be computed ({e}) -- likely too few "
              f"observations relative to parameters, given small n.")

    print("\nInterpretation: p < 0.05 on either test suggests heteroskedasticity")
    print("(non-constant error variance), which would bias standard OLS")
    print("standard errors. With n this small, these tests themselves have")
    print("low power -- a non-significant result here is weak evidence of")
    print("homoskedasticity, not strong confirmation of it.")

    return {"bp_pvalue": bp_pvalue, "white_pvalue": white_pvalue}


# ----------------------------------------------------------------------
# 3. Refit with robust standard errors
# ----------------------------------------------------------------------
def refit_robust(game_days, het_results):
    section("3. Refit with HC1/HC3 robust standard errors")

    if het_results is None:
        print("Skipped (no baseline model).")
        return

    formula = f"{PRIMARY_SENTIMENT_COL} ~ points + plus_minus + C(win_loss) + C(home_away)"

    print("Refitting with HC3 (better suited to small samples than HC1):\n")
    model_hc3 = smf.ols(formula=formula, data=game_days).fit(cov_type="HC3")
    print(model_hc3.summary())

    print("\nFor comparison, HC1:\n")
    model_hc1 = smf.ols(formula=formula, data=game_days).fit(cov_type="HC1")
    print(model_hc1.summary())


# ----------------------------------------------------------------------
# 4. ARMA time series model on the full daily sentiment series
# ----------------------------------------------------------------------
def run_arma(df):
    section("4. ARMA time series model on daily sentiment")

    series = df.set_index("comment_date")[PRIMARY_SENTIMENT_COL]
    n_missing = series.isna().sum()
    print(f"Daily sentiment series: n={len(series)} days, {n_missing} missing")

    if n_missing > 0:
        print("Interpolating missing days (linear) for ARMA fitting -- ARMA")
        print("requires a complete series. Days with 0 comments produce NaN")
        print("averages; interpolation is a simplification worth noting in")
        print("the writeup rather than treating as ground truth.")
        series = series.interpolate(method="linear")

    # ARMA(1,1) as a starting point -- but with only 31 observations,
    # jointly estimating both an AR and MA term is often poorly identified.
    # Fit it, but explicitly check for warning signs (non-convergence, an AR
    # coefficient pinned near the +/-1 stationarity boundary, or a wildly
    # oversized MA standard error) rather than trusting the output blindly.
    # Fall back to a simpler AR(1)-only model if any of those show up.
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            arma_result = ARIMA(series, order=(1, 0, 1)).fit()
            converged = not any("failed to converge" in str(w.message) for w in caught)

        ar_coef = arma_result.params.get("ar.L1")
        ma_coef = arma_result.params.get("ma.L1")
        ma_se = arma_result.bse.get("ma.L1")
        boundary_issue = ar_coef is not None and abs(abs(ar_coef) - 1.0) < 0.01
        unstable_ma = (
            ma_se is not None and ma_coef not in (None, 0)
            and abs(ma_se / ma_coef) > 2
        )

        print(arma_result.summary())

        if not converged or boundary_issue or unstable_ma:
            reasons = []
            if not converged:
                reasons.append("optimizer did not converge")
            if boundary_issue:
                reasons.append(f"AR coefficient ({ar_coef:.4f}) sits at the stationarity boundary")
            if unstable_ma:
                reasons.append(f"MA coefficient's standard error ({ma_se:.3f}) dwarfs its estimate ({ma_coef:.3f})")
            print("\nWARNING: ARMA(1,1) shows signs of an unreliable fit --")
            print("  " + "; ".join(reasons))
            print("This ARMA(1,1) result should NOT be reported as-is. Falling back")
            print("to a simpler AR(1) model, which has fewer parameters to estimate")
            print("and is more likely to be reliably identified at n=31:\n")
            ar1_result = ARIMA(series, order=(1, 0, 0)).fit()
            print(ar1_result.summary())

    except Exception as e:
        print(f"ARMA(1,1) failed to fit entirely ({e}). Trying AR(1) instead:\n")
        try:
            ar1_result = ARIMA(series, order=(1, 0, 0)).fit()
            print(ar1_result.summary())
        except Exception as e2:
            print(f"AR(1) also failed ({e2}). Series may be too short/unstable")
            print("for any ARMA-family model at this sample size.")


# ----------------------------------------------------------------------
# 5. Event study: sentiment before vs. after the MVP announcement
# ----------------------------------------------------------------------
def run_event_study(df):
    section("5. Event study: pre vs. post May 4 MVP announcement")

    pre = df[~df["post_mvp_announcement"]][PRIMARY_SENTIMENT_COL].dropna()
    post = df[df["post_mvp_announcement"]][PRIMARY_SENTIMENT_COL].dropna()

    print(f"Pre-announcement days:  n={len(pre)}, mean={pre.mean():.4f}, std={pre.std():.4f}")
    print(f"Post-announcement days: n={len(post)}, mean={post.mean():.4f}, std={post.std():.4f}")

    if len(pre) < 2 or len(post) < 2:
        print("Too few observations on one side to run a t-test.")
        return

    t_stat, p_value = scipy_stats.ttest_ind(post, pre, equal_var=False)

    print(f"\nWelch's t-test (unequal variance): t={t_stat:.4f}, p={p_value:.4f}")
    print("NOTE: pre-period is only 3 days (May 1-3) -- this comparison has")
    print("very little statistical power on the 'before' side. Treat any")
    print("result here as illustrative rather than a confirmed effect.")


def main():
    df = load_data()
    print(f"Loaded {len(df)} days from daily_sentiment_and_performance")
    print(f"Primary sentiment column: {PRIMARY_SENTIMENT_COL}")

    model, game_days = run_ols(df)
    het_results = test_heteroskedasticity(model, game_days)
    refit_robust(game_days, het_results)
    run_arma(df)
    run_event_study(df)

    section("Done")
    print("Next steps: review coefficient signs/significance, decide whether")
    print("ARMA order needs adjustment, and write up the small-n caveats")
    print("explicitly rather than letting p-values stand alone.")


if __name__ == "__main__":
    main()