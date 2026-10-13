"""
A/B Testing & Statistical Significance Analysis.

Run from any working directory with:
    python ab_test_analysis.py

The CSV is resolved relative to this script. SRM is tested only when the
experiment's intended allocation ratio is supplied from documentation.
CUPED is demonstrated on a separate synthetic experiment; the marketing data
does not contain a genuine pre-treatment covariate.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import chisquare
from statsmodels.formula.api import logit
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportions_ztest, proportion_effectsize

RANDOM_SEED = 42
DATA_PATH = Path(__file__).resolve().parent / "marketing_AB.csv"
REQUIRED_COLUMNS = {"user_id", "test_group", "converted", "total_ads",
                    "most_ads_day", "most_ads_hour"}


def load_data(path=None):
    """Load and validate the CSV; column names are normalized to snake_case."""
    csv_path = Path(path) if path is not None else DATA_PATH
    if not csv_path.is_absolute() and path is not None:
        csv_path = Path(__file__).resolve().parent / csv_path
    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip().replace(" ", "_") for c in df.columns]
    return validate_data(df)


def validate_data(df):
    """Validate one-row-per-user experiment data and return a normalized copy."""
    df = df.copy()
    df.columns = [str(c).strip().replace(" ", "_") for c in df.columns]
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    if df[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("Required columns contain missing values.")
    if df["user_id"].duplicated().any():
        raise ValueError("Duplicate user_id values found; expected one row per user.")
    groups = set(df["test_group"].unique())
    if groups != {"ad", "psa"}:
        raise ValueError(f"Expected test_group values 'ad' and 'psa'; got {sorted(groups)}.")
    converted = pd.to_numeric(df["converted"], errors="coerce")
    if converted.isna().any() or not converted.isin([0, 1]).all():
        raise ValueError("converted must contain only binary values 0 and 1.")
    df["converted"] = converted.astype(int)
    total_ads = pd.to_numeric(df["total_ads"], errors="coerce")
    if total_ads.isna().any() or (total_ads < 0).any():
        raise ValueError("total_ads must contain numeric, non-negative values.")
    df["total_ads"] = total_ads
    return df


def check_srm(df, expected_ratio=None, alpha=0.001):
    """Run SRM chi-square test only with a documented (ad, psa) allocation ratio."""
    df = validate_data(df)
    observed = df["test_group"].value_counts().reindex(["ad", "psa"]).to_numpy()
    result = {"status": "not_tested", "chi2": None, "p_value": None,
              "srm_detected": None, "observed": dict(zip(["ad", "psa"], map(int, observed)))}
    print("=== Sample Ratio Mismatch (SRM) ===")
    print(f"Observed group counts: {result['observed']}")
    if expected_ratio is None:
        print("SRM not tested: provide the documented allocation ratio (ad, psa).\n")
        return result
    if (len(expected_ratio) != 2 or any(not np.isfinite(x) or x <= 0 for x in expected_ratio)
            or not np.isclose(sum(expected_ratio), 1.0)):
        raise ValueError("expected_ratio must be two positive proportions summing to 1.")
    expected = np.asarray(expected_ratio, dtype=float) * observed.sum()
    chi2, p_value = chisquare(f_obs=observed, f_exp=expected)
    result.update(status="tested", chi2=float(chi2), p_value=float(p_value),
                  srm_detected=bool(p_value < alpha),
                  expected=dict(zip(["ad", "psa"], map(float, expected))))
    print(f"Documented expected ratio (ad, psa): {tuple(expected_ratio)}")
    print(f"Chi-square={chi2:.4f}, p-value={p_value:.6g}, mismatch detected={p_value < alpha}\n")
    return result


def run_ztest(df):
    df = validate_data(df)
    ad = df.loc[df.test_group == "ad", "converted"].to_numpy()
    psa = df.loc[df.test_group == "psa", "converted"].to_numpy()
    counts, sizes = np.array([ad.sum(), psa.sum()]), np.array([len(ad), len(psa)])
    z_stat, p_value = proportions_ztest(counts, sizes)
    p_ad, p_psa = ad.mean(), psa.mean()
    lift = p_ad - p_psa
    relative_lift = lift / p_psa * 100 if p_psa else np.nan
    se = np.sqrt(p_ad * (1-p_ad)/len(ad) + p_psa * (1-p_psa)/len(psa))
    ci = (float(lift - 1.96*se), float(lift + 1.96*se))
    print("=== Two-proportion Z-test ===")
    print(f"ad conversion: {p_ad:.4%}; psa conversion: {p_psa:.4%}")
    print(f"Absolute lift: {lift:.4%}; relative lift: {relative_lift:.2f}%")
    print(f"z={z_stat:.4f}; p={p_value:.6g}; approximate 95% CI: [{ci[0]:.4%}, {ci[1]:.4%}]\n")
    return {"p_ad": float(p_ad), "p_psa": float(p_psa), "abs_lift": float(lift),
            "rel_lift": float(relative_lift), "z_stat": float(z_stat),
            "p_value": float(p_value), "ci": ci}


def bootstrap_validation(df, n_boot=10000, seed=RANDOM_SEED):
    """Reproducible percentile bootstrap for difference in binary conversion rates."""
    df = validate_data(df)
    if not isinstance(n_boot, int) or n_boot < 1:
        raise ValueError("n_boot must be a positive integer.")
    ad = df.loc[df.test_group == "ad", "converted"].to_numpy()
    psa = df.loc[df.test_group == "psa", "converted"].to_numpy()
    rng = np.random.default_rng(seed)
    # For binary data, a resampled conversion count is exactly Binomial(n, observed rate).
    ad_means = rng.binomial(len(ad), ad.mean(), size=n_boot) / len(ad)
    psa_means = rng.binomial(len(psa), psa.mean(), size=n_boot) / len(psa)
    diffs = ad_means - psa_means
    ci = tuple(map(float, np.percentile(diffs, [2.5, 97.5])))
    result = {"mean_lift": float(diffs.mean()), "ci": ci,
              "pct_positive": float((diffs > 0).mean()), "n_boot": n_boot, "seed": seed}
    print(f"=== Bootstrap validation ({n_boot:,} resamples; seed={seed}) ===")
    print(f"Mean lift: {result['mean_lift']:.4%}; percentile 95% CI: [{ci[0]:.4%}, {ci[1]:.4%}]")
    print(f"Resamples with positive lift: {result['pct_positive']:.2%}\n")
    return result


def business_impact(df):
    """Descriptive extrapolation versus PSA baseline; not proof of causal attribution."""
    df = validate_data(df)
    ad = df[df.test_group == "ad"]
    psa = df[df.test_group == "psa"]
    expected = len(ad) * psa.converted.mean()
    estimate = float(ad.converted.sum() - expected)
    print("=== Illustrative business impact ===")
    print(f"Ad users: {len(ad):,}; observed conversions: {int(ad.converted.sum()):,}")
    print(f"Expected at PSA conversion rate: {expected:,.1f}")
    print(f"Estimated incremental conversions vs PSA baseline: {estimate:,.1f}")
    print("Interpret only if randomization, measurement, and causal assumptions are credible.\n")
    return {"estimated_incremental_conversions": estimate}


def power_analysis(df):
    """Retrospective MDE calculation; it is not an independent test of the observed effect."""
    df = validate_data(df)
    ad = df.loc[df.test_group == "ad", "converted"]
    psa = df.loc[df.test_group == "psa", "converted"]
    n_ad, n_psa = len(ad), len(psa)
    p_psa, p_ad = float(psa.mean()), float(ad.mean())
    effect = NormalIndPower().solve_power(effect_size=None, nobs1=n_psa,
        alpha=0.05, power=0.80, ratio=n_ad/n_psa)
    mde_p_ad = brentq(lambda p: proportion_effectsize(p, p_psa) - effect,
                      p_psa + 1e-12, 0.999999)
    mde = float(mde_p_ad - p_psa)
    print("=== Retrospective minimum detectable effect (MDE) ===")
    print(f"Sample sizes: ad={n_ad:,}; psa={n_psa:,}; baseline={p_psa:.4%}")
    print(f"Approximate MDE at 80% power and alpha=.05: {mde:.4%}")
    print(f"Observed lift: {p_ad-p_psa:.4%}. This retrospective calculation is descriptive, not proof of power.\n")
    return {"mde_abs": mde, "observed_lift": p_ad-p_psa}


def synthetic_cuped_demo(n=20000, seed=RANDOM_SEED):
    """Demonstrate CUPED on synthetic randomized data with a genuine simulated pre-period covariate."""
    if not isinstance(n, int) or n < 100 or n % 2:
        raise ValueError("n must be an even integer of at least 100.")
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)  # simulated pre-treatment metric, generated independently of outcomes
    treatment = np.zeros(n, dtype=int)
    treatment[rng.permutation(n)[:n//2]] = 1
    probability = 1 / (1 + np.exp(-(-3.3 + 0.9*x + 0.25*treatment)))
    y = rng.binomial(1, probability)
    theta = float(np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1))
    adjusted = y - theta*(x-x.mean())
    raw_vars, adj_vars = [], []
    for group in (0, 1):
        raw_vars.append(float(np.var(y[treatment == group], ddof=1)))
        adj_vars.append(float(np.var(adjusted[treatment == group], ddof=1)))
    raw_se = np.sqrt(sum(v/(n//2) for v in raw_vars))
    adj_se = np.sqrt(sum(v/(n//2) for v in adj_vars))
    result = {"n": n, "seed": seed, "theta": theta,
              "variance_reduction_pct": float((1-np.mean(adj_vars)/np.mean(raw_vars))*100),
              "raw_se": float(raw_se), "adjusted_se": float(adj_se),
              "se_reduction_pct": float((1-adj_se/raw_se)*100),
              "raw_lift": float(y[treatment == 1].mean()-y[treatment == 0].mean()),
              "adjusted_lift": float(adjusted[treatment == 1].mean()-adjusted[treatment == 0].mean())}
    print("=== Synthetic CUPED demonstration (not the marketing dataset) ===")
    print(f"n={n:,}; seed={seed}; theta={theta:.4f}")
    print(f"Mean within-group variance reduction: {result['variance_reduction_pct']:.1f}%")
    print(f"Treatment-effect SE reduction: {result['se_reduction_pct']:.1f}%")
    print("Synthetic result only; real CUPED requires a pre-treatment covariate collected before assignment.\n")
    return result


def regression_and_hte(df):
    """Exploratory logistic models only; total_ads is post-treatment exposure, not a baseline covariate."""
    df = validate_data(df)
    data = df.copy()
    data["treatment"] = (data.test_group == "ad").astype(int)
    print("=== Exploratory regression / HTE ===")
    print("Caution: total_ads is post-treatment exposure; coefficients/interactions are not clean causal estimates.")
    additive = logit("converted ~ treatment + total_ads", data=data).fit(disp=0)
    interaction = logit("converted ~ treatment * total_ads", data=data).fit(disp=0)
    print("Additive model coefficients and p-values:")
    print(additive.summary2().tables[1][["Coef.", "P>|z|"]])
    print("\nTreatment-by-exposure interaction coefficients and p-values:")
    print(interaction.summary2().tables[1][["Coef.", "P>|z|"]])
    print("A nonsignificant interaction is not evidence that effects are equal across exposure levels.\n")
    return {"additive_model": additive, "interaction_model": interaction}


def main():
    df = load_data()
    check_srm(df)  # ratio intentionally omitted: documented design allocation is unavailable
    run_ztest(df)
    bootstrap_validation(df)
    business_impact(df)
    power_analysis(df)
    synthetic_cuped_demo()  # separate synthetic example, never run on the real outcome data
    regression_and_hte(df)  # exploratory only


if __name__ == "__main__":
    main()
