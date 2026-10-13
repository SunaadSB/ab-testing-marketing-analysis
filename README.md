# A/B Testing & Statistical Significance Analysis

Reproducible analysis of the [Marketing A/B Testing dataset](https://www.kaggle.com/datasets/faviovaz/marketing-ab-testing) by Faviovaz (588,101 user rows), comparing an ad condition with a PSA control.

## Research question

Did the observed conversion rate differ between users assigned to the ad and PSA conditions? Causal interpretation depends on valid assignment, measurement, and experiment assumptions.

## Methods

1. **Validation:** required columns, binary outcomes, group labels, missing values, numeric non-negative ad counts, and one row per user.
2. **Two-proportion Z-test:** conversion rates, absolute/relative lift, p-value, and approximate 95% confidence interval.
3. **Bootstrap:** reproducible percentile interval with a local random generator and seed. Binomial draws are equivalent to resampling binary outcomes with replacement.
4. **Illustrative business impact:** ad conversions compared with the PSA baseline rate. This is conditional on causal assumptions, not proof that the difference was caused by ads.
5. **Retrospective MDE:** approximate detectable effect for observed sample sizes at 80% power and alpha 0.05; not independent proof of adequate power.
6. **Synthetic CUPED:** separate generated randomized experiment with a simulated pre-treatment covariate. The marketing data contains no genuine pre-treatment covariate, so no CUPED benefit is claimed for it.
7. **Exploratory regression/HTE:** logistic models with `total_ads` and a treatment interaction. Since `total_ads` is post-treatment exposure, these models do not provide clean causal effect-modification estimates. A nonsignificant interaction does not prove equal effects.

## Key findings

On the current dataset, the ad group has a **2.555%** observed conversion rate versus **1.785%** for the PSA group. The observed absolute difference is **+0.769 percentage points** (relative lift **+43.09%**). The two-proportion Z-test gives **p = 1.705 × 10⁻¹³**, with an approximate 95% confidence interval for the absolute lift of **[0.595, 0.943] percentage points**; the bootstrap interval is similar at **[0.596, 0.939] percentage points**.

These are observed group differences, not automatically causal effects. The intended assignment ratio is undocumented, so sample-ratio mismatch (SRM) remains **untested**. The dataset also lacks cost/ROI and guardrail metrics, and the CUPED example is synthetic.

## Data and SRM assumptions

The expected input `marketing_AB.csv` is in the repository root. Column names are normalized from spaces to underscores.

SRM is **not tested** unless the experiment's intended allocation ratio is supplied from the experiment documentation. The ratio must not be inferred from observed counts. For example, only if documented:

```python
check_srm(df, expected_ratio=(0.96, 0.04))
```

## Run locally

Python 3.10+ is recommended.

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python ab_test_analysis.py
python -m pytest
jupyter notebook ab_test_analysis.ipynb
```

Run the notebook from the repository root. The script resolves its default CSV path relative to its own file. Notebook cells use shared functions from `ab_test_analysis.py`.

## Repository layout

```
.
├── marketing_AB.csv
├── ab_test_analysis.py
├── ab_test_analysis.ipynb
├── requirements.txt
├── tests/
│   └── test_analysis.py
├── .gitignore
└── README.md
```

## Interpretation guardrails

- Statistical significance does not establish ROI or net benefit; cost and guardrail metrics are unavailable.
- The Z-test confidence interval is a normal-approximation interval.
- SRM is not tested without a documented allocation ratio.
- CUPED results are synthetic teaching output, not results from the marketing data.
- Regression/HTE is exploratory because ad exposure is post-treatment.
- MDE is a retrospective descriptive calculation, not independent evidence that the experiment was well-powered.
