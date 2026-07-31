"""
statistical_modeling.py

Statistical analysis of SGA sentiment vs. game performance and
the May 21 MVP announcement. Pulls from the daily_sentiment_and_performance
view (see sql/03_aggregation_queries.sql) and runs:

1. OLS: does game performance predict same-day sentiment?
2. Breusch-Pagan / White heteroskedasticity tests on the OLS residuals
3. Refit with HC1/HC3 robust standard errors
4. ARMA time series model on the daily sentiment series (with automatic
   fallback to AR(1) if ARMA(1,1) shows signs of an unreliable fit)
5. Event study: sentiment before vs. after the May 21 MVP announcement,
   PLUS a playoff-controlled version isolating the announcement effect
   from the confound of playoffs starting around the same time
6. Power analysis, computed directly via the noncentral F distribution
7. Bivariate correlations
8. Win/loss MARGIN interaction: does the sentiment effect of plus_minus
   differ between wins and losses (blowout vs close games)?

SCALE NOTE: originally written against n=31 days / n=11 games (May 2015
pilot). Running against the full 2024-25 season (273 days, ~99 games)
is a meaningfully better-powered design.

POWER ANALYSIS FIX: the original version used statsmodels' FTestPower
with effect_size=0.35, intended as Cohen's f^2. At n=99 the resulting
power estimate (6.2%) was implausibly low, suggesting a units mismatch
in how FTestPower interprets effect_size. This version computes power
directly via scipy's noncentral F distribution using lambda = f^2 * n
(the standard Cohen 1988 noncentrality formula), unambiguous and
independent of any library's parameter convention.

EVENT STUDY CONFOUND: SGA's MVP announcement (May 21, 2025) landed in
the middle of the Thunder's playoff run, not a quiet stretch -- a naive
pre/post comparison conflates "effect of the announcement" with "effect
of playoff basketball generating more intense discourse." The
playoff-controlled event study compares pre- vs. post-announcement
sentiment ONLY within the playoff window to isolate the announcement's
own effect. PLAYOFF_START_DATE is an estimate -- verify against actual
player_stats data before treating as precise.

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

PG_USER = os.getenv("POSTGRES_USER", "sga_admin")
PG_PASSWORD = os.getenv("POSTGRES_PASSWORD")
PG_HOST = os.getenv("POSTGRES_HOST", "localhost")
PG_PORT = os.getenv("POSTGRES_PORT", "5432")
PG_DB = os.getenv("POSTGRES_DB", "sga_sentiment")

PRIMARY_SENTIMENT_COL = "llm_avg_score"
MVP_ANNOUNCEMENT_DATE = "2025-05-21"

# ESTIMATE -- verify against actual data before treating as precise.
PLAYOFF_START_DATE = "2025-04-19"

BLOWOUT_THRESHOLD = 15


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


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def run_ols(df):
    section("1. OLS: game performance -> same-day sentiment")

    game_days = df[df["is_game_day"]].copy()
    n = len(game_days)
    print(f"Game days available for regression: n={n}")
    if n < 5:
        print("Too few game days to run a meaningful regression. Skipping.")
        return None, game_days, None

    formula = f"{PRIMARY_SENTIMENT_COL} ~ points + plus_minus + C(win_loss) + C(home_away)"
    model = smf.ols(formula=formula, data=game_days).fit()
    print(model.summary())
    print_significance(model)

    return model, game_days, formula


def run_ols_next_day(df):
    section("1b. OLS: game performance -> NEXT-DAY sentiment")

    print("MOTIVATION: NBA games tip off 7-10pm local time. A same-day")
    print("comparison implicitly assumes most reaction happens before that")
    print("day's comment volume is posted, which likely isn't true -- a lot")
    print("of genuine post-game reaction lands on the following calendar day.")
    print("Separately, created_at here is computed from raw UTC timestamps,")
    print("not adjusted to US time zones -- a 7-10pm ET tipoff is ~11pm-2am")
    print("UTC, so some same-night reaction may already fall on the next UTC")
    print("calendar day even before accounting for people commenting all day.")
    print("This section re-runs the same OLS, but matches each game to the")
    print("FOLLOWING day's sentiment instead of the same day's.\n")

    sentiment_by_date = df.set_index("comment_date")[PRIMARY_SENTIMENT_COL]

    game_days = df[df["is_game_day"]].copy()
    game_days["next_day"] = game_days["comment_date"] + pd.Timedelta(days=1)
    game_days["next_day_sentiment"] = game_days["next_day"].map(sentiment_by_date)

    n_available = game_days["next_day_sentiment"].notna().sum()
    n_total = len(game_days)
    print(f"Game days with a next-day sentiment value available: {n_available}/{n_total}")
    if n_available < n_total:
        print(f"({n_total - n_available} missing -- likely game dates at the very")
        print("end of the season window with no following day in the dataset.)\n")

    fit_data = game_days.dropna(subset=["next_day_sentiment"])
    if len(fit_data) < 5:
        print("Too few observations with a valid next-day match. Skipping.")
        return None, None, None

    formula = "next_day_sentiment ~ points + plus_minus + C(win_loss) + C(home_away)"
    model = smf.ols(formula=formula, data=fit_data).fit()
    print(model.summary())
    print_significance(model)

    print("\nCompare this R^2 and coefficients against section 1's same-day")
    print("version.")

    return model, fit_data, formula


def run_ols_combined_window(df):
    section("1c. OLS: game performance -> COMBINED same-day + next-day sentiment")

    print("MOTIVATION: 1b showed next-day sentiment alone captures much more")
    print("signal than same-day alone. This variant instead pools BOTH days'")
    print("comments into a single weighted-average sentiment score per game --")
    print("weighted by each day's comment COUNT (not a flat 50/50 average of")
    print("the two daily means), so a day with more sampled comments")
    print("contributes proportionally more to the combined estimate.\n")

    n_col = "llm_n_comments"
    sentiment_by_date = df.set_index("comment_date")[PRIMARY_SENTIMENT_COL]
    n_by_date = df.set_index("comment_date")[n_col]

    game_days = df[df["is_game_day"]].copy()
    game_days["next_day"] = game_days["comment_date"] + pd.Timedelta(days=1)
    game_days["next_day_sentiment"] = game_days["next_day"].map(sentiment_by_date)
    game_days["next_day_n"] = game_days["next_day"].map(n_by_date)

    same_n = game_days[n_col].fillna(0)
    next_n = game_days["next_day_n"].fillna(0)
    same_s = game_days[PRIMARY_SENTIMENT_COL]
    next_s = game_days["next_day_sentiment"]

    total_n = same_n + next_n
    # Weighted average; guard against total_n == 0 (no comments on either day)
    with np.errstate(invalid="ignore", divide="ignore"):
        combined = (same_s.fillna(0) * same_n + next_s.fillna(0) * next_n) / total_n
    combined[total_n == 0] = np.nan
    game_days["combined_sentiment"] = combined

    fit_data = game_days.dropna(subset=["combined_sentiment"])
    n_available = len(fit_data)
    print(f"Game days with a valid combined sentiment value: {n_available}/{len(game_days)}\n")

    if n_available < 5:
        print("Too few observations. Skipping.")
        return None, None, None

    formula = "combined_sentiment ~ points + plus_minus + C(win_loss) + C(home_away)"
    model = smf.ols(formula=formula, data=fit_data).fit()
    print(model.summary())
    print_significance(model)

    print("\nCompare R^2 here against 1 (same-day only, R^2 lower) and 1b")
    print("(next-day only) -- if combining days doesn't improve on 1b alone,")
    print("next-day is likely capturing the real signal and same-day comments")
    print("are mostly just adding noise to the estimate rather than useful")
    print("information.")

    return model, fit_data, formula


def print_significance(model):
    print("\n--- Significance at alpha=0.10 (90% confidence), for reference ---")
    for name, p in model.pvalues.items():
        if name == "Intercept":
            continue
        if p < 0.05:
            print(f"  {name}: p={p:.4f} -- significant at alpha=0.05 (95% CI)")
        elif p < 0.10:
            print(f"  {name}: p={p:.4f} -- significant at alpha=0.10 (90% CI) ONLY; treat as exploratory")
        else:
            print(f"  {name}: p={p:.4f} -- not significant at either threshold")


def test_heteroskedasticity(model, label="model"):
    section(f"2/3. Heteroskedasticity tests (Breusch-Pagan, White) -- {label}")

    if model is None:
        print("No model to test.")
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
        print(f"White test could not be computed ({e}).")

    print("\np < 0.05 on either test suggests heteroskedasticity, which would")
    print("bias standard OLS standard errors.")

    return {"bp_pvalue": bp_pvalue, "white_pvalue": white_pvalue}


def refit_robust(data, formula, label="model"):
    section(f"2/3. Refit with HC1/HC3 robust standard errors -- {label}")

    if data is None or formula is None:
        print("Skipped (no baseline model).")
        return

    print("HC3:\n")
    print(smf.ols(formula=formula, data=data).fit(cov_type="HC3").summary())

    print("\nHC1 (for comparison):\n")
    print(smf.ols(formula=formula, data=data).fit(cov_type="HC1").summary())


def run_arma(df):
    section("4. ARMA time series model on daily sentiment")

    series = df.set_index("comment_date")[PRIMARY_SENTIMENT_COL]
    n_missing = series.isna().sum()
    print(f"Daily sentiment series: n={len(series)} days, {n_missing} missing")

    if n_missing > 0:
        print("Interpolating missing days (linear) for ARMA fitting.")
        series = series.interpolate(method="linear")

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
                reasons.append(f"MA coef SE ({ma_se:.3f}) dwarfs its estimate ({ma_coef:.3f})")
            print("\nWARNING: ARMA(1,1) unreliable -- " + "; ".join(reasons))
            print("Falling back to AR(1):\n")
            print(ARIMA(series, order=(1, 0, 0)).fit().summary())
        else:
            print(f"\nAR(1) coef = {ar_coef:.3f}: today's sentiment strongly predicts")
            print("tomorrow's -- real day-to-day persistence, even though (per")
            print("section 1) it doesn't track same-day box scores.")

    except Exception as e:
        print(f"ARMA(1,1) failed ({e}). Trying AR(1):\n")
        try:
            print(ARIMA(series, order=(1, 0, 0)).fit().summary())
        except Exception as e2:
            print(f"AR(1) also failed ({e2}).")


def run_event_study(df):
    section(f"5a. Event study (NAIVE, full sample): pre vs. post {MVP_ANNOUNCEMENT_DATE}")

    pre = df[~df["post_mvp_announcement"]][PRIMARY_SENTIMENT_COL].dropna()
    post = df[df["post_mvp_announcement"]][PRIMARY_SENTIMENT_COL].dropna()

    print(f"Pre:  n={len(pre)}, mean={pre.mean():.4f}, std={pre.std():.4f}")
    print(f"Post: n={len(post)}, mean={post.mean():.4f}, std={post.std():.4f}")

    if len(pre) < 2 or len(post) < 2:
        print("Too few observations on one side.")
        return

    t_stat, p_value = scipy_stats.ttest_ind(post, pre, equal_var=False)
    print(f"\nWelch's t-test: t={t_stat:.4f}, p={p_value:.4f}")

    print("\nCAUTION: CONFOUNDED. Post-period is almost entirely playoff games;")
    print("pre-period mixes regular season and non-game days. See 5b.")


def run_event_study_playoff_controlled(df):
    section(f"5b. Event study (PLAYOFF-CONTROLLED): pre vs. post {MVP_ANNOUNCEMENT_DATE}")

    playoff_df = df[df["comment_date"] >= pd.Timestamp(PLAYOFF_START_DATE)]
    print(f"Playoff window: {PLAYOFF_START_DATE} onward (ESTIMATE)")
    print(f"Days in window: {len(playoff_df)}\n")

    pre = playoff_df[~playoff_df["post_mvp_announcement"]][PRIMARY_SENTIMENT_COL].dropna()
    post = playoff_df[playoff_df["post_mvp_announcement"]][PRIMARY_SENTIMENT_COL].dropna()

    pre_mean = pre.mean() if len(pre) else float("nan")
    post_mean = post.mean() if len(post) else float("nan")
    print(f"Pre (playoffs, pre-MVP):  n={len(pre)}, mean={pre_mean:.4f}")
    print(f"Post (playoffs, post-MVP): n={len(post)}, mean={post_mean:.4f}")

    if len(pre) < 2 or len(post) < 2:
        print("\nToo few observations on one side -- check PLAYOFF_START_DATE.")
        return

    t_stat, p_value = scipy_stats.ttest_ind(post, pre, equal_var=False)
    print(f"\nWelch's t-test (playoff-controlled): t={t_stat:.4f}, p={p_value:.4f}")
    print("\nCompare to 5a: if similarly significant, stronger evidence of a")
    print("genuine announcement effect. If it shrinks, the naive result was")
    print("likely mostly the regular-season-to-playoffs transition.")


def power_given_r2(r2, n, k, alpha=0.05):
    df1, df2 = k, n - k - 1
    f2 = r2 / (1 - r2)
    ncp = f2 * n
    f_crit = scipy_stats.f.ppf(1 - alpha, df1, df2)
    return 1 - scipy_stats.ncf.cdf(f_crit, df1, df2, ncp)


def r2_needed_for_power(target_power, n, k, alpha=0.05):
    lo, hi = 1e-6, 0.999
    for _ in range(60):
        mid = (lo + hi) / 2
        if power_given_r2(mid, n, k, alpha) < target_power:
            lo = mid
        else:
            hi = mid
    return hi


def run_power_analysis(game_days, n_predictors=4):
    section("6. Power analysis: what could this study design detect?")

    n = len(game_days)
    df_denom = n - n_predictors - 1
    print(f"n={n}, {n_predictors} predictors, residual df={df_denom}")

    if df_denom < 1:
        print("Not enough residual df.")
        return

    r2_needed = r2_needed_for_power(0.80, n, n_predictors)
    print(f"\nMinimum detectable R^2 at 80% power, alpha=0.05: {r2_needed:.3f} ({r2_needed:.1%})")
    print(f"Equivalent correlation: {r2_needed**0.5:.3f}")

    large_r2 = 0.35 / 1.35
    print(f"\nPower to detect a 'large' effect (f^2=0.35, R^2={large_r2:.1%}): "
          f"{power_given_r2(large_r2, n, n_predictors):.1%}")
    medium_r2 = 0.15 / 1.15
    print(f"Power to detect a 'medium' effect (f^2=0.15, R^2={medium_r2:.1%}): "
          f"{power_given_r2(medium_r2, n, n_predictors):.1%}")

    print("\nINTERPRETATION:", end=" ")
    if r2_needed > 0.5:
        print("could only detect an almost deterministic relationship.")
    else:
        print("reasonable power for moderate-to-large effects -- compare section")
        print(f"1's actual R^2 against the {r2_needed:.1%} threshold above.")


def run_bivariate_correlations(data, sentiment_col=None, label="same-day (section 7)"):
    sentiment_col = sentiment_col or PRIMARY_SENTIMENT_COL
    section(f"Bivariate correlations (single predictor at a time) -- {label}")

    print(f"n={len(data)}\n")
    for col in ["points", "plus_minus", "rebounds", "assists"]:
        if col not in data.columns:
            continue
        paired = data[[col, sentiment_col]].dropna()
        if len(paired) < 4:
            print(f"{col}: too few observations, skipping.")
            continue
        r, p_value = scipy_stats.pearsonr(paired[col], paired[sentiment_col])
        if abs(r) >= 0.9999:
            print(f"{col}: r={r:.3f} -- near-perfect, likely unstable.")
            continue
        n_pairs = len(paired)
        z = np.arctanh(r)
        se = 1 / np.sqrt(n_pairs - 3)
        ci_low, ci_high = np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)
        print(f"{col} vs. {sentiment_col}: r={r:.3f}, p={p_value:.3f}, "
              f"95% CI=[{ci_low:.3f}, {ci_high:.3f}], n={n_pairs}")


def run_margin_interaction(data, sentiment_col=None, label="same-day (section 8)"):
    sentiment_col = sentiment_col or PRIMARY_SENTIMENT_COL
    section(f"Win/loss margin interaction -- {label}")

    print(f"n={len(data)}")
    print(f"Does the sentiment effect of plus_minus differ for wins vs. losses? "
          f"(outcome: {sentiment_col})\n")

    formula = f"{sentiment_col} ~ C(win_loss) * plus_minus + C(home_away)"
    model = smf.ols(formula=formula, data=data).fit()
    print(model.summary())

    for term in [n for n in model.pvalues.index if ":" in n]:
        p, coef = model.pvalues[term], model.params[term]
        sig = "significant (p<0.05)" if p < 0.05 else ("suggestive (p<0.10)" if p < 0.10 else "not significant")
        print(f"\nInteraction {term}: coef={coef:.4f}, p={p:.4f} -- {sig}")

    print(f"\n--- Blowout (|margin|>={BLOWOUT_THRESHOLD}) vs close ---")
    gd = data.copy()
    gd["margin_type"] = np.where(gd["plus_minus"].abs() >= BLOWOUT_THRESHOLD, "blowout", "close")
    print(gd.groupby(["win_loss", "margin_type"])[sentiment_col].agg(["mean", "std", "count"]))


def main():
    df = load_data()
    print(f"Loaded {len(df)} days from daily_sentiment_and_performance")
    print(f"Primary sentiment column: {PRIMARY_SENTIMENT_COL}")

    model, game_days, formula = run_ols(df)
    test_heteroskedasticity(model, "same-day (section 1)")
    refit_robust(game_days, formula, "same-day (section 1)")

    model_next, data_next, formula_next = run_ols_next_day(df)
    test_heteroskedasticity(model_next, "next-day (section 1b)")
    refit_robust(data_next, formula_next, "next-day (section 1b)")

    model_combined, data_combined, formula_combined = run_ols_combined_window(df)
    test_heteroskedasticity(model_combined, "combined window (section 1c)")
    refit_robust(data_combined, formula_combined, "combined window (section 1c)")

    run_arma(df)
    run_event_study(df)
    run_event_study_playoff_controlled(df)
    run_power_analysis(game_days)

    section("7/8. Same-day vs. next-day: bivariate correlations and margin interaction")
    print("Given the 1b discovery that next-day sentiment carries far more signal")
    print("than same-day, both tests below are run on BOTH timings for comparison.")

    run_bivariate_correlations(game_days, PRIMARY_SENTIMENT_COL, "same-day (7a)")
    if data_next is not None:
        run_bivariate_correlations(data_next, "next_day_sentiment", "next-day (7b)")

    if game_days is not None and len(game_days) >= 5:
        run_margin_interaction(game_days, PRIMARY_SENTIMENT_COL, "same-day (8a)")
    if data_next is not None and len(data_next) >= 5:
        run_margin_interaction(data_next, "next_day_sentiment", "next-day (8b)")

    section("Done")


if __name__ == "__main__":
    main()