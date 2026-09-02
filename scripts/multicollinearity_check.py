"""
multicollinearity_check.py

Diagnoses and addresses the collinearity between C(win_loss), points, and
plus_minus in the next-day sentiment regression (see statistical_modeling.py
section 1b, extended with HAC standard errors in newey_west_comparison.py).

WHY THIS SCRIPT EXISTS: plus_minus (point differential) is what determines
win_loss's sign in basketball -- a team's plus_minus is positive if and only
if it won -- and points scored is one of the two terms that make up
plus_minus (plus_minus = points_scored - points_allowed). Running all three
in the same OLS specification means they compete for credit on what is
largely the same underlying signal ("how much did the team win or lose
by"), which inflates standard errors and makes individual coefficients
unstable. That instability is visible in section 1b's own output: plus_minus's
significance verdict flips depending on which standard error type is used
(see newey_west_comparison.py's "next-day" comparison), while win_loss and
points stay significant throughout -- exactly the symptom collinearity
produces.

This script:
  1. Computes Variance Inflation Factors (VIF) for the full four-predictor
     specification, to measure the collinearity directly rather than just
     asserting it from the win_loss/plus_minus relationship.
  2. Refits three specifications side by side -- full, drop-plus_minus, and
     drop-win_loss -- each with HAC(maxlags=3) standard errors (the lag
     length newey_west_comparison.py settled on), to see how each
     predictor's coefficient and significance move depending on what else
     is in the model.
  3. Recomputes VIF for each reduced specification, to confirm the drop
     actually fixes the collinearity rather than just hiding it.

CONVENTION NOTE: VIF > 5 is a commonly used (but not universal) threshold
for "worth addressing"; VIF > 10 is the more conservative "clearly a
problem" threshold. Neither is a hard statistical law -- treat these as a
judgment call, not a pass/fail test.

Usage:
    python scripts/multicollinearity_check.py

Requires the same Postgres container / .env setup as statistical_modeling.py
(reads from the daily_sentiment_and_performance view).
"""
import os

import pandas as pd
import patsy
import statsmodels.formula.api as smf
from dotenv import load_dotenv
from sqlalchemy import create_engine
from statsmodels.stats.outliers_influence import variance_inflation_factor

load_dotenv()

PG_USER = os.getenv("POSTGRES_USER", "sga_admin")
PG_PASSWORD = os.getenv("POSTGRES_PASSWORD")
PG_HOST = os.getenv("POSTGRES_HOST", "localhost")
PG_PORT = os.getenv("POSTGRES_PORT", "5432")
PG_DB = os.getenv("POSTGRES_DB", "sga_sentiment")

PRIMARY_SENTIMENT_COL = "llm_avg_score"
HAC_MAXLAGS = 3  # matches the rule-of-thumb lag settled on in newey_west_comparison.py


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


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
    """Mirrors build_next_day_frame() in newey_west_comparison.py."""
    sentiment_by_date = df.set_index("comment_date")[PRIMARY_SENTIMENT_COL]
    game_days = df[df["is_game_day"]].copy()
    game_days["next_day"] = game_days["comment_date"] + pd.Timedelta(days=1)
    game_days["next_day_sentiment"] = game_days["next_day"].map(sentiment_by_date)
    return game_days.dropna(subset=["next_day_sentiment"]).sort_values("comment_date")


def print_vif(data, formula, label):
    """Builds the model's design matrix via the same formula OLS would use,
    and reports VIF for every predictor except the intercept (VIF isn't a
    meaningful diagnostic for the intercept itself)."""
    _, X = patsy.dmatrices(formula, data, return_type="dataframe")
    predictor_cols = [c for c in X.columns if c != "Intercept"]

    print(f"\n--- VIF: {label} ---")
    print(f"formula: {formula}\n")
    rows = []
    for col in predictor_cols:
        vif = variance_inflation_factor(X.values, X.columns.get_loc(col))
        flag = "SEVERE (>10)" if vif > 10 else ("moderate (>5)" if vif > 5 else "OK")
        rows.append({"predictor": col, "VIF": round(vif, 2), "flag": flag})
    result = pd.DataFrame(rows).set_index("predictor")
    print(result)
    return result


def fit_hac(data, formula):
    return smf.ols(formula=formula, data=data).fit(
        cov_type="HAC", cov_kwds={"maxlags": HAC_MAXLAGS}
    )


def compare_specifications(data, specs):
    """Fits each formula in `specs` (label -> formula) with
    HAC(maxlags=HAC_MAXLAGS) standard errors, and builds one table of each
    predictor's coefficient and p-value across all specs -- so it's visible
    at a glance whether e.g. points' effect holds up once plus_minus is no
    longer in the model competing for the same variance, or whether it was
    only "significant" in the full model because of the collinearity."""
    fits = {label: fit_hac(data, formula) for label, formula in specs.items()}

    all_predictors = sorted({
        name for fit in fits.values() for name in fit.params.index if name != "Intercept"
    })

    rows = []
    for predictor in all_predictors:
        row = {"predictor": predictor}
        for label, fit in fits.items():
            if predictor in fit.params.index:
                row[f"{label}_coef"] = round(fit.params[predictor], 4)
                row[f"{label}_p"] = round(fit.pvalues[predictor], 4)
            else:
                row[f"{label}_coef"] = None
                row[f"{label}_p"] = None
        rows.append(row)

    print("\n=== Coefficient / p-value comparison across specifications (HAC) ===")
    with pd.option_context("display.width", 160):
        print(pd.DataFrame(rows).set_index("predictor"))

    print("\n=== Model fit comparison ===")
    for label, fit in fits.items():
        print(f"{label}: R^2={fit.rsquared:.3f}, adj R^2={fit.rsquared_adj:.3f}, "
              f"AIC={fit.aic:.1f}, BIC={fit.bic:.1f}")

    return fits


def main():
    section("Multicollinearity check: next-day sentiment specification")

    df = load_data()
    next_day = build_next_day_frame(df)
    print(f"n={len(next_day)} game days with a valid next-day sentiment match")

    full_formula = "next_day_sentiment ~ points + plus_minus + C(win_loss) + C(home_away)"
    print_vif(next_day, full_formula, "full specification (all 4 predictors)")

    specs = {
        "full": full_formula,
        "drop_plus_minus": "next_day_sentiment ~ points + C(win_loss) + C(home_away)",
        "drop_win_loss": "next_day_sentiment ~ points + plus_minus + C(home_away)",
    }
    compare_specifications(next_day, specs)

    section("VIF after dropping the collinear term")
    print_vif(next_day, specs["drop_plus_minus"], "drop_plus_minus")
    print_vif(next_day, specs["drop_win_loss"], "drop_win_loss")

    section("Reading this output")
    print("If points' coefficient and p-value barely move between 'full' and")
    print("'drop_plus_minus', and VIF drops substantially, that's a specification")
    print("worth reporting as primary: it keeps the interpretable win/loss story,")
    print("keeps points as a continuous predictor, and isn't distorted by the")
    print("mechanical link between plus_minus and win_loss. If instead points' or")
    print("win_loss's significance DEPENDED on plus_minus being in the model,")
    print("that's worth flagging explicitly in the writeup rather than picking")
    print("one spec silently.")


if __name__ == "__main__":
    main()
