# A/B Testing & Statistical Significance Analysis

End-to-end A/B test analysis on a real-world marketing dataset, extended with production-grade experimentation techniques (SRM validation, power analysis, CUPED, and heterogeneous treatment effect testing).

## Problem

A company ran an experiment comparing two conditions:
- **`ad`** — users shown a product advertisement
- **`psa`** — control group shown a generic public service announcement instead

**Question:** Does showing ads increase conversion rate, and is the effect statistically and practically significant?

**Dataset:** [Marketing A/B Testing](https://www.kaggle.com/datasets/faviovaz/marketing-ab-testing) (Kaggle, Faviovaz) — 588,101 users, columns: `user id`, `test group`, `converted`, `total ads`, `most ads day`, `most ads hour`.

## Approach

1. **Sample Ratio Mismatch (SRM) check** — chi-square goodness-of-fit test confirming the observed group split matched the intended allocation ratio, ruling out randomization/logging bugs before trusting any downstream result.
2. **Hypothesis testing** — two-sample Z-test for proportions, with absolute/relative lift and 95% confidence intervals.
3. **Bootstrap validation** — 10,000-iteration resampling to cross-check the Z-test result without relying on normality assumptions.
4. **Business impact translation** — converting the statistical lift into incremental conversions.
5. **Power analysis / MDE** — confirming the sample size was sufficient to detect the observed effect (post-hoc, since this is historical data — in production this step precedes the experiment to size it correctly).
6. **CUPED variance reduction** — demonstrating how a pre-experiment covariate shrinks variance and tightens confidence intervals.
7. **Covariate adjustment + heterogeneous treatment effects (HTE)** — logistic regression controlling for ad exposure, plus an interaction term testing whether the treatment effect varies by exposure level.

## Results

| Metric | Value |
|---|---|
| `ad` conversion rate | 2.55% |
| `psa` conversion rate | 1.79% |
| Absolute lift | 0.77 pp |
| Relative lift | 43.1% |
| p-value (Z-test) | < 0.0001 |
| 95% CI | [0.60%, 0.94%] |
| Bootstrap 95% CI | [0.59%, 0.94%] (matches Z-test) |
| SRM check | No mismatch detected (p = 0.9998) |
| MDE at observed sample size | 0.26 pp (well below observed 0.77 pp lift → well-powered) |
| Extra conversions attributable to ads | ~4,343 |
| CUPED variance reduction (simulated strong pre-period signal) | 28.5% (11% narrower CI) |
| HTE by ad exposure (interaction term p-value) | 0.37 — not significant |

**Conclusion:** Ads produced a statistically significant and practically meaningful lift in conversion (43% relative, p < 0.0001), robust across two independent test methods (Z-test and bootstrap) and confirmed well-powered via MDE analysis. The treatment effect was stable across exposure levels (no significant interaction with `total_ads`), meaning the lift wasn't concentrated among heavy- or light-exposure users.

## Limitations

- **No cost or guardrail metrics.** The dataset has no ad spend, unsubscribe, or complaint data, so ROI and unintended negative effects couldn't be assessed — a full business decision would need these.
- **CUPED uses a simulated pre-experiment covariate.** No genuine pre-period metric exists in this dataset (all columns reflect in-experiment behavior). The pre-period engagement score was simulated to demonstrate the CUPED mechanism correctly; production use requires a real historical covariate (e.g., prior purchase activity).
- **SRM expected ratio was inferred**, not sourced from documented experiment design, since the dataset doesn't include the intended allocation split.

## Project Structure

```
ab_test_project/
├── data/
│   └── marketing_AB.csv
├── notebooks/
│   └── ab_test_analysis.ipynb   # Executed notebook with all outputs
├── src/
│   └── ab_test_analysis.py      # Same pipeline as a runnable script
└── README.md
```

## Running It

```bash
pip install pandas numpy scipy statsmodels
python src/ab_test_analysis.py
```

or open `notebooks/ab_test_analysis.ipynb` in Jupyter.
