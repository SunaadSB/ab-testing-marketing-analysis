import numpy as np
import pandas as pd
import pytest

from ab_test_analysis import (
    bootstrap_validation, business_impact, check_srm, run_ztest,
    synthetic_cuped_demo, validate_data,
)


def fixture_df():
    return pd.DataFrame({
        "user_id": range(1, 17),
        "test_group": ["ad"] * 8 + ["psa"] * 8,
        "converted": [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
        "total_ads": [1, 2, 1, 3, 1, 0, 2, 0, 1, 0, 2, 1, 0, 0, 3, 1],
        "most_ads_day": ["Monday"] * 16, "most_ads_hour": [12] * 16,
    })


def test_column_normalization():
    df = fixture_df().rename(columns={"user_id": "user id", "test_group": "test group"})
    assert "user_id" in validate_data(df)


def test_missing_column_rejected():
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_data(fixture_df().drop(columns=["total_ads"]))


@pytest.mark.parametrize("bad", [2, -1, "bad"])
def test_nonbinary_conversion_rejected(bad):
    df = fixture_df()
    df.loc[0, "converted"] = bad
    with pytest.raises(ValueError, match="binary"):
        validate_data(df)


def test_missing_values_rejected():
    df = fixture_df()
    df.loc[0, "converted"] = np.nan
    with pytest.raises(ValueError, match="missing"):
        validate_data(df)


def test_duplicate_user_rejected():
    df = fixture_df()
    df.loc[1, "user_id"] = df.loc[0, "user_id"]
    with pytest.raises(ValueError, match="Duplicate user_id"):
        validate_data(df)


def test_invalid_group_rejected():
    df = fixture_df()
    df.loc[df.test_group == "psa", "test_group"] = "ad"
    with pytest.raises(ValueError, match="Expected test_group"):
        validate_data(df)


def test_ztest_known_lift():
    result = run_ztest(fixture_df())
    assert result["p_ad"] == pytest.approx(0.5)
    assert result["p_psa"] == pytest.approx(0.25)
    assert result["abs_lift"] == pytest.approx(0.25)
    assert result["rel_lift"] == pytest.approx(100.0)


def test_srm_not_tested_without_ratio():
    result = check_srm(fixture_df())
    assert result["status"] == "not_tested"
    assert result["p_value"] is None
    assert result["srm_detected"] is None


def test_srm_with_documented_ratio():
    result = check_srm(fixture_df(), expected_ratio=(0.5, 0.5))
    assert result["status"] == "tested"
    assert 0 <= result["p_value"] <= 1
    assert result["srm_detected"] is False


def test_invalid_srm_ratio_rejected():
    with pytest.raises(ValueError, match="summing to 1"):
        check_srm(fixture_df(), expected_ratio=(0.9, 0.9))


def test_bootstrap_reproducible():
    first = bootstrap_validation(fixture_df(), n_boot=500, seed=123)
    second = bootstrap_validation(fixture_df(), n_boot=500, seed=123)
    assert first == second


@pytest.mark.parametrize("n_boot", [0, -1, 1.5])
def test_invalid_bootstrap_count_rejected(n_boot):
    with pytest.raises(ValueError, match="positive integer"):
        bootstrap_validation(fixture_df(), n_boot=n_boot)


def test_business_impact_named_as_estimate():
    result = business_impact(fixture_df())
    assert result["estimated_incremental_conversions"] == pytest.approx(2.0)


def test_synthetic_cuped_reproducible():
    first = synthetic_cuped_demo(n=2000, seed=17)
    second = synthetic_cuped_demo(n=2000, seed=17)
    assert first == second
    assert np.isfinite(first["theta"])
    assert np.isfinite(first["adjusted_se"])


@pytest.mark.parametrize("n", [99, 101, 0])
def test_invalid_cuped_sample_size_rejected(n):
    with pytest.raises(ValueError, match="even integer"):
        synthetic_cuped_demo(n=n)


def test_infinite_total_ads_rejected():
    df = fixture_df()
    df.loc[0, "total_ads"] = np.inf
    with pytest.raises(ValueError, match="finite"):
        validate_data(df)
