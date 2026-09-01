"""
newey_west_comparison.py

Extends statistical_modeling.py's OLS results with Newey-West (HAC) robust
standard errors, to check whether the day-to-day sentiment autocorrelation
found by run_arma() (AR(1) coefficient = 0.935) changes any significance
conclusions from the classical or HC3-robust versions of the regression.

Newey-West / HAC ("Heteroskedasticity and Autocorrelation Consistent")
standard errors relax BOTH the constant-variance assumption (like HC3
already does) AND the no-autocorrelation assumption, by allowing errors
from nearby observations (here: nearby CALENDAR DAYS) to be correlated up
to a chosen number of lags, with more distant lags down-weighted (a
Bartlett kernel). Practically, statsmodels implements the whole thing as
`.fit(cov_type="HAC", cov_kwds={"maxlags": L})` -- no need to hand-code the
sandwich estimator -- but L (how many lags of correlation to account for)
has to be chosen deliberately, and the data has to actually be in time
order for "nearby lags" to mean anything.

Choosing maxlags: Newey & West's own (1994) rule of thumb is
    L = floor(4 * (T/100)^(2/9))
At T=99 (game days) that comes out to ~3. But the ARMA(1,1) result
specifically found AR(1)-shaped persistence -- correlation concentrated at
lag 1, decaying geometrically after that -- so this script checks BOTH
L=1 (targeted at the specific persistence actually found) and the
rule-of-thumb L, since under-correcting (too few lags) leaves bias in the
standard errors, while over-correcting (too many lags) just adds noise
without fixing anything further.

Usage:
    python scripts/newey_west_comparison.py

Requires the same Postgres container / .env setup as statistical_modeling.py
(reads from the daily_sentiment_and_performance view).
"""
import os

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()

PG_USER = os.getenv("POSTGRES_USER", "sga_admin")
PG_PASSWORD = os.getenv("POSTGRES_PASSWORD")
PG_HOST = os.getenv("POSTGRES_HOST", "localhost")
PG_PORT = os.getenv("POSTGRES_PORT", "5432")
PG_DB = os.getenv("POSTGRES_DB", "sga_sentiment")

PRIMARY_SENTIMENT_COL = "llm_avg_score"


def load_data():
    engine = create_engine(
        f"postgresql+psycopg2://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DB}"
    )
    df = pd.read_sql(
        "SELECT * FROM daily_sentiment_and_performance ORDER BY comment_date",
        engine,
    )
    df["comment_date"] = pd.to_datetime(df["comment_date"])
    return df


def build_next_day_frame(df):
    """Mirrors run_ols_next_day() in statistical_modeling.py: matches each
    game day to the FOLLOWING calendar day's sentiment instead of the same
    day's."""
    sentiment_by_date = df.set_index("comment_date")[PRIMARY_SENTIMENT_COL]
    game_days = df[df["is_game_day"]].copy()
    game_days["next_day"] = game_days["comment_date"] + pd.Timedelta(days=1)
    game_days["next_day_sentiment"] = game_days["next_day"].map(sentiment_by_date)
    return game_days.dropna(subset=["next_day_sentiment"]).sort_values("comment_date")


def newey_west_lag_rule_of_thumb(n_obs):
    """Newey & West (1994) suggested lag length: floor(4*(T/100)^(2/9))."""
    return int(np.floor(4 * (n_obs / 100) ** (2 / 9)))


def compare_covariance_types(data, formula, label, extra_lags=()):
    """
    Fits the SAME OLS specification multiple ways -- classical (assumes
    i.i.d. errors), HC3 (heteroskedasticity-robust, still assumes
    independence), and HAC/Newey-West at one or more lag lengths (relaxes
    BOTH heteroskedasticity and independence) -- and prints/returns a
    comparison of each predictor's p-value and significance flag across
    all of them.

    IMPORTANT: HAC assumes `data` is already sorted in the time order the
    lags are meant to be counted over. Passing unsorted rows silently
    computes a meaningless correction -- there's no error, just a wrong
    answer, so the caller sorting by date first is load-bearing, not
    cosmetic.
    """
    n = len(data)
    lags_to_try = list(dict.fromkeys(
        [1, newey_west_lag_rule_of_thumb(n), *extra_lags]
    ))  # de-duplicate while preserving order (e.g. if rule-of-thumb == 1)

    fits = {
        "classical": smf.ols(formula=formula, data=data).fit(),
        "HC3": smf.ols(formula=formula, data=data).fit(cov_type="HC3"),
    }
    for L in lags_to_try:
        fits[f"HAC(maxlags={L})"] = smf.ols(formula=formula, data=data).fit(
            cov_type="HAC", cov_kwds={"maxlags": L}
        )

    predictor_names = [p for p in fits["classical"].pvalues.index if p != "Intercept"]
    rows = []
    for predictor in predictor_names:
        row = {"predictor": predictor}
        for cov_label, fit in fits.items():
            p = fit.pvalues[predictor]
            row[f"{cov_label}_p"] = p
            row[f"{cov_label}_sig@0.05"] = p < 0.05
        rows.append(row)

    result = pd.DataFrame(rows).set_index("predictor")

    print(f"\n=== Covariance-type comparison: {label} (n={n}) ===")
    with pd.option_context("display.width", 160):
        print(result)

    # The number that actually matters isn't "did the p-value move a bit,"
    # it's "did any predictor's SIGNIFICANCE VERDICT flip" -- that's the
    # case that would change what the project can honestly claim.
    sig_cols = [c for c in result.columns if c.endswith("_sig@0.05")]
    flips = result[sig_cols].nunique(axis=1) > 1
    if flips.any():
        print("\n  Significance verdict CHANGES across covariance types for:")
        print("   ", ", ".join(flips[flips].index.tolist()))
    else:
        print("\n  No predictor's significance verdict (sig/not-sig at 0.05) "
              "changes across classical / HC3 / HAC. The finding is robust "
              "to correcting for the autocorrelation ARMA found.")

    return result


def main():
    df = load_data()

    same_day = df[df["is_game_day"]].copy().sort_values("comment_date")
    next_day = build_next_day_frame(df)

    compare_covariance_types(
        same_day,
        f"{PRIMARY_SENTIMENT_COL} ~ points + plus_minus + C(win_loss) + C(home_away)",
        "same-day (section 1)",
    )
    compare_covariance_types(
        next_day,
        "next_day_sentiment ~ points + plus_minus + C(win_loss) + C(home_away)",
        "next-day (section 1b)",
    )


if __name__ == "__main__":
    main()
