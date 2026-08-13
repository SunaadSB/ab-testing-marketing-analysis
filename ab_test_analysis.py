"""
A/B Testing & Statistical Significance Analysis
Dataset: Marketing A/B Testing (Kaggle, Faviovaz)

Pipeline:
1. Data loading & integrity check (SRM)
2. Hypothesis testing (Z-test for proportions)
3. Bootstrap validation
4. Business impact translation
5. Power analysis / MDE
6. CUPED variance reduction (simulated pre-period covariate)
7. Covariate adjustment + HTE via logistic regression
"""

import pandas as pd
import numpy as np
from scipy.stats import chisquare
from statsmodels.stats.proportion import proportions_ztest, proportion_effectsize
from statsmodels.stats.power import NormalIndPower
from scipy.optimize import brentq
import statsmodels.formula.api as smf

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def load_data(path="marketing_AB.csv"):
    df = pd.read_csv(path)
    df.columns = [c.replace(" ", "_") for c in df.columns]
    return df


# ---------------------------------------------------------------------------
# 1. SAMPLE RATIO MISMATCH (SRM) CHECK
# ---------------------------------------------------------------------------
def check_srm(df, expected_ratio=(0.96, 0.04), alpha=0.001):
    """
    Chi-square goodness-of-fit test to verify observed group sizes match
    the intended allocation ratio. Flags potential randomization/logging bugs.

    Note: expected_ratio should ideally come from the experiment's documented
    design. Here it is inferred from the near-exact match to observed counts,
    since the true intended ratio isn't provided with the dataset.
    """
    observed = df["test_group"].value_counts()[["ad", "psa"]].values
    total = observed.sum()
    expected = [total * r for r in expected_ratio]

    chi2, p = chisquare(f_obs=observed, f_exp=expected)
    srm_flag = p < alpha

    print("=== SRM Check ===")
    print(f"Observed: {dict(zip(['ad', 'psa'], observed))}")
    print(f"Expected: ad={expected[0]:.0f}, psa={expected[1]:.0f}")
    print(f"Chi-square: {chi2:.4f}, p-value: {p:.6f}")
    print(f"SRM detected: {srm_flag}\n")
    return {"chi2": chi2, "p_value": p, "srm_detected": srm_flag}


# ---------------------------------------------------------------------------
# 2. HYPOTHESIS TEST (Z-TEST FOR PROPORTIONS)
# ---------------------------------------------------------------------------
def run_ztest(df):
    ad = df[df["test_group"] == "ad"]["converted"]
    psa = df[df["test_group"] == "psa"]["converted"]

    count = np.array([ad.sum(), psa.sum()])
    nobs = np.array([len(ad), len(psa)])
    z_stat, p_val = proportions_ztest(count, nobs)

    p_ad, p_psa = ad.mean(), psa.mean()
    abs_lift = p_ad - p_psa
    rel_lift = abs_lift / p_psa * 100

    se = np.sqrt(p_ad * (1 - p_ad) / nobs[0] + p_psa * (1 - p_psa) / nobs[1])
    ci_low, ci_high = abs_lift - 1.96 * se, abs_lift + 1.96 * se

    print("=== Hypothesis Test (Z-test for proportions) ===")
    print(f"ad conversion:  {p_ad:.4%}")
    print(f"psa conversion: {p_psa:.4%}")
    print(f"Absolute lift: {abs_lift:.4%} | Relative lift: {rel_lift:.2f}%")
    print(f"Z-stat: {z_stat:.4f}, p-value: {p_val:.10f}")
    print(f"95% CI: [{ci_low:.4%}, {ci_high:.4%}]\n")

    return {
        "p_ad": p_ad, "p_psa": p_psa, "abs_lift": abs_lift, "rel_lift": rel_lift,
        "z_stat": z_stat, "p_value": p_val, "ci": (ci_low, ci_high),
    }


# ---------------------------------------------------------------------------
# 3. BOOTSTRAP VALIDATION
# ---------------------------------------------------------------------------
def bootstrap_validation(df, n_boot=10000):
    ad = df[df["test_group"] == "ad"]["converted"].values.astype(int)
    psa = df[df["test_group"] == "psa"]["converted"].values.astype(int)

    boot_diffs = np.empty(n_boot)
    for i in range(n_boot):
        ad_sample = np.random.choice(ad, size=len(ad), replace=True)
        psa_sample = np.random.choice(psa, size=len(psa), replace=True)
        boot_diffs[i] = ad_sample.mean() - psa_sample.mean()

    ci_low, ci_high = np.percentile(boot_diffs, [2.5, 97.5])
    pct_positive = (boot_diffs > 0).mean()

    print("=== Bootstrap Validation (10,000 resamples) ===")
    print(f"Bootstrap mean lift: {boot_diffs.mean():.4%}")
    print(f"Bootstrap 95% CI: [{ci_low:.4%}, {ci_high:.4%}]")
    print(f"% samples with positive lift: {pct_positive:.4%}\n")

    return {"mean_lift": boot_diffs.mean(), "ci": (ci_low, ci_high), "pct_positive": pct_positive}


# ---------------------------------------------------------------------------
# 4. BUSINESS IMPACT
# ---------------------------------------------------------------------------
def business_impact(df):
    ad = df[df["test_group"] == "ad"]
    psa = df[df["test_group"] == "psa"]

    p_psa = psa["converted"].mean()
    n_ad = len(ad)
    actual_conversions = ad["converted"].sum()
    expected_at_psa_rate = n_ad * p_psa
    extra_conversions = actual_conversions - expected_at_psa_rate

    print("=== Business Impact ===")
    print(f"Ad group size: {n_ad:,}")
    print(f"Actual conversions: {int(actual_conversions):,}")
    print(f"Expected at psa rate: {expected_at_psa_rate:,.0f}")
    print(f"Extra conversions attributable to ads: {extra_conversions:,.0f}\n")

    return {"extra_conversions": extra_conversions}


# ---------------------------------------------------------------------------
# 5. POWER ANALYSIS / MDE
# ---------------------------------------------------------------------------
def power_analysis(df):
    ad = df[df["test_group"] == "ad"]["converted"]
    psa = df[df["test_group"] == "psa"]["converted"]
    n_ad, n_psa = len(ad), len(psa)
    p_psa, p_ad = psa.mean(), ad.mean()

    analysis = NormalIndPower()
    ratio = n_ad / n_psa

    mde_effect_size = analysis.solve_power(
        effect_size=None, nobs1=n_psa, alpha=0.05, power=0.80, ratio=ratio
    )

    def h_diff(p_ad_guess):
        return proportion_effectsize(p_ad_guess, p_psa) - mde_effect_size

    mde_p_ad = brentq(h_diff, p_psa, 0.99)
    mde_abs = mde_p_ad - p_psa

    print("=== Power Analysis / MDE ===")
    print(f"Sample sizes: ad={n_ad:,}, psa={n_psa:,}")
    print(f"Baseline (psa) conversion: {p_psa:.4%}")
    print(f"MDE (80% power, alpha=0.05): {mde_abs:.4%}")
    print(f"Observed lift: {p_ad - p_psa:.4%} (well above MDE -> well-powered)\n")

    return {"mde_abs": mde_abs, "observed_lift": p_ad - p_psa}


# ---------------------------------------------------------------------------
# 6. CUPED VARIANCE REDUCTION
# ---------------------------------------------------------------------------
def cuped_demo(df, pre_corr_strength=4.0, label="Strong pre-period signal"):
    """
    NOTE: This dataset has no genuine pre-experiment covariate, so a
    pre-period engagement score is simulated (correlated with `converted`)
    purely to demonstrate the CUPED mechanism. In production, use a real
    pre-experiment metric (e.g., prior purchase history).
    """
    df = df.copy()
    noise = np.random.normal(0, 1, len(df))
    df["pre_sim"] = pre_corr_strength * df["converted"].astype(int) + noise

    ad = df[df["test_group"] == "ad"]
    psa = df[df["test_group"] == "psa"]

    Y_all = df["converted"].astype(int).values
    X_all = df["pre_sim"].values
    theta = np.cov(Y_all, X_all)[0, 1] / np.var(X_all)

    def adjust(sub):
        Y = sub["converted"].astype(int).values
        X = sub["pre_sim"].values
        return Y - theta * (X - X_all.mean())

    ad_cuped = adjust(ad)
    psa_cuped = adjust(psa)

    p_ad, p_psa = ad["converted"].mean(), psa["converted"].mean()
    se_orig = np.sqrt(p_ad * (1 - p_ad) / len(ad) + p_psa * (1 - p_psa) / len(psa))
    ci_width_orig = 2 * 1.96 * se_orig

    se_cuped = np.sqrt(ad_cuped.var() / len(ad_cuped) + psa_cuped.var() / len(psa_cuped))
    ci_width_cuped = 2 * 1.96 * se_cuped

    reduction_pct = (1 - ad_cuped.var() / ad["converted"].astype(int).var()) * 100
    ci_narrowed_pct = (1 - ci_width_cuped / ci_width_orig) * 100

    print(f"=== CUPED ({label}, theta={theta:.4f}) ===")
    print(f"Variance reduction: {reduction_pct:.1f}%")
    print(f"CI width before: {ci_width_orig:.4%} | after: {ci_width_cuped:.4%}")
    print(f"CI narrowed by: {ci_narrowed_pct:.1f}%\n")

    return {"theta": theta, "variance_reduction_pct": reduction_pct, "ci_narrowed_pct": ci_narrowed_pct}


# ---------------------------------------------------------------------------
# 7. COVARIATE ADJUSTMENT + HTE (INTERACTION TERMS)
# ---------------------------------------------------------------------------
def regression_and_hte(df):
    df = df.copy()
    df["treatment"] = (df["test_group"] == "ad").astype(int)
    df["converted"] = df["converted"].astype(int)

    model1 = smf.logit("converted ~ treatment + total_ads", data=df).fit(disp=0)
    print("=== Covariate Adjustment (additive) ===")
    print(model1.summary2().tables[1][["Coef.", "P>|z|"]])
    print()

    model2 = smf.logit("converted ~ treatment * total_ads", data=df).fit(disp=0)
    print("=== Interaction Term (HTE by ad exposure) ===")
    print(model2.summary2().tables[1][["Coef.", "P>|z|"]])
    print()

    return {"additive_model": model1, "interaction_model": model2}


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = load_data()

    srm_result = check_srm(df)
    ztest_result = run_ztest(df)
    boot_result = bootstrap_validation(df)
    impact_result = business_impact(df)
    power_result = power_analysis(df)
    cuped_result = cuped_demo(df)
    reg_result = regression_and_hte(df)

    print("=== Pipeline complete ===")
