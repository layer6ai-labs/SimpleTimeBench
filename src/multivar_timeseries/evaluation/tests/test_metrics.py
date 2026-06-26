import numpy as np
import pytest
from multivar_timeseries.evaluation.metrics import (
    _get_input_means,
    mean_abs_deviation_label,
    mean_abs_deviation_forecast,
    MeanAttractionRatio,
    path_absolute_difference_label,
    path_absolute_difference_forecast,
    PathVolatilityRatio,
    MeanBiasRatio,
    PersistenceSkillRatio,
    MeanSkillRatio,
    NormalizedMeanBias,
    signed_deviation_stats,
    persistence_relative_error_stats,
    mean_baseline_relative_error_stats,
)


def test_get_input_means_1d():
    data = {"input": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])}
    means = _get_input_means(data)
    expected = np.array([[2.0], [5.0]])
    np.testing.assert_allclose(means, expected)


def test_get_input_means_2d_multivariate():
    # Multivariate case: shape (batch, features, time)
    # series 0: [[1,2,3], [4,5,6]] -> feature means [2.0, 5.0]
    # series 1: [[7,8,9], [10,11,12]] -> feature means [8.0, 11.0]
    data = {
        "input": np.array(
            [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]]
        )
    }
    means = _get_input_means(data)
    expected = np.array([[2.0, 5.0], [8.0, 11.0]])
    np.testing.assert_allclose(means, expected)


def test_get_input_means_context_length():
    data = {"input": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])}
    means = _get_input_means(data, context_length=2)
    expected = np.array([[2.5], [5.5]])
    np.testing.assert_allclose(means, expected)


def test_get_input_means_with_nans():
    data = {"input": np.array([[1.0, np.nan, 3.0], [4.0, 5.0, 6.0]])}
    means = _get_input_means(data)
    expected = np.array([[2.0], [5.0]])
    np.testing.assert_allclose(means, expected)


def test_mean_abs_deviation_label():
    data = {"input": np.array([[1.0, 2.0, 3.0]]), "label": np.array([[4.0, 5.0]])}
    dev = mean_abs_deviation_label(data)
    expected = np.array([[2.0, 3.0]])
    np.testing.assert_allclose(dev, expected)


def test_mean_abs_deviation_forecast():
    data = {
        "input": np.array([[1.0, 2.0, 3.0]]),  # mean = 2.0
        "0.5": np.array([[1.0, 5.0]]),
    }
    # |1-2|, |5-2| = 1.0, 3.0
    dev = mean_abs_deviation_forecast(data, forecast_type="0.5")
    expected = np.array([[1.0, 3.0]])
    np.testing.assert_allclose(dev, expected)


def test_mar_metric():
    data = {
        "input": np.array([[1.0, 2.0, 3.0]]),
        "label": np.array([[4.0, 6.0]]),
        "0.5": np.array([[3.0, 3.0]]),
    }
    mar_call = MeanAttractionRatio(forecast_type="0.5")
    mar_instance = mar_call()
    mar_instance.update(data)
    result = mar_instance.get()
    assert pytest.approx(result) == 3.0
    assert "MeanAttractionRatio" in mar_instance.name


def test_path_absolute_difference_label():
    data = {"input": np.array([[10.0, 20.0, 30.0]]), "label": np.array([[35.0, 45.0]])}
    diffs = path_absolute_difference_label(data)
    expected = np.array([[5.0, 10.0]])
    np.testing.assert_allclose(diffs, expected)


def test_path_absolute_difference_forecast():
    data = {"input": np.array([[10.0, 20.0, 30.0]]), "0.5": np.array([[35.0, 45.0]])}
    diffs = path_absolute_difference_forecast(data, forecast_type="0.5")
    expected = np.array([[5.0, 10.0]])
    np.testing.assert_allclose(diffs, expected)


def test_path_volatility_ratio():
    data = {
        "input": np.array([[10.0]]),
        "label": np.array([[20.0, 30.0]]),
        "0.5": np.array([[15.0, 25.0]]),
    }
    pvr_call = PathVolatilityRatio(forecast_type="0.5")
    pvr_instance = pvr_call()
    pvr_instance.update(data)
    result = pvr_instance.get()
    assert pytest.approx(result) == 1.3333333333
    assert "PathVolatilityRatio" in pvr_instance.name


def test_mbr_metric():
    data = {
        "input": np.array([[10.0]]),
        "label": np.array([[12.0]]),
        "0.5": np.array([[11.0]]),
    }
    mbr_call = MeanBiasRatio(forecast_type="0.5")
    mbr_metric = mbr_call()
    mbr_metric.update(data)
    assert mbr_metric.get() == 1.0
    assert "MeanBiasRatio" in mbr_metric.name


def test_mbr_not_biased():
    data = {
        "input": np.array([[10.0]]),
        "label": np.array([[12.0]]),
        "0.5": np.array([[13.0]]),
    }
    mbr_call = MeanBiasRatio(forecast_type="0.5")
    mbr_metric = mbr_call()
    mbr_metric.update(data)
    assert mbr_metric.get() == 0.0


def test_persistence_skill_ratio():
    data = {
        "input": np.array([[10.0, 20.0]]),
        "label": np.array([[25.0, 30.0]]),
        "0.5": np.array([[22.0, 28.0]]),
    }
    psr_call = PersistenceSkillRatio(forecast_type="0.5")
    psr_instance = psr_call()
    psr_instance.update(data)
    result = psr_instance.get()
    assert pytest.approx(result) == 0.3333333333
    assert "PSR" in psr_instance.name


def test_mean_skill_ratio():
    data = {
        "input": np.array([[10.0, 20.0, 30.0]]),
        "label": np.array([[25.0, 35.0]]),
        "0.5": np.array([[24.0, 36.0]]),
    }
    msr_call = MeanSkillRatio(forecast_type="0.5")
    msr_instance = msr_call()
    msr_instance.update(data)
    result = msr_instance.get()
    assert pytest.approx(result) == 0.1
    assert "MSR" in msr_instance.name


def test_persistence_relative_error_stats():
    data = {
        "input": np.array([[10.0, 20.0]]),
        "label": np.array([[25.0]]),
        "0.5": np.array([[22.0]]),
    }
    m_err, p_err = persistence_relative_error_stats(data, forecast_type="0.5")
    assert m_err[0, 0] == 3.0  # |25-22|
    assert p_err[0, 0] == 5.0  # |25-20|


def test_mean_baseline_relative_error_stats():
    data = {
        "input": np.array([[10.0, 20.0, 30.0]]),
        "label": np.array([[25.0]]),
        "0.5": np.array([[24.0]]),
    }
    m_err, b_err = mean_baseline_relative_error_stats(data, forecast_type="0.5")
    assert m_err[0, 0] == 1.0  # |25-24|
    assert b_err[0, 0] == 5.0  # |25-20|


def test_wrappers():
    from multivar_timeseries.evaluation.metrics import (
        MEAN_DEVIATION_LABEL,
        MEAN_DEVIATION_FORECAST,
        PATH_VOLATILITY_LABEL,
        PATH_VOLATILITY_FORECAST,
    )

    data = {
        "input": np.array([[10.0, 20.0, 30.0]]),
        "label": np.array([[35.0]]),
        "0.5": np.array([[25.0]]),
    }
    # MEAN_DEVIATION_LABEL: mean=20, label=35 -> |35-20|=15
    assert MEAN_DEVIATION_LABEL()().stat(data) == 15.0
    # MEAN_DEVIATION_FORECAST: mean=20, 0.5=25 -> |25-20|=5
    assert MEAN_DEVIATION_FORECAST(forecast_type="0.5")().stat(data) == 5.0
    # PATH_VOLATILITY_LABEL: last=30, label=35 -> |35-30|=5
    assert PATH_VOLATILITY_LABEL()().stat(data) == 5.0
    # PATH_VOLATILITY_FORECAST: last=30, 0.5=25 -> |25-30|=5
    assert PATH_VOLATILITY_FORECAST(forecast_type="0.5")().stat(data) == 5.0


def test_normalized_mean_bias():
    data = {
        "label": np.array([[110.0, 110.0]]),  # Mean = 110
        "0.5": np.array([[100.0, 100.0]]),  # Error = 10, 10. Mean = 10
    }
    # NMB = 10 / 110 = 0.0909090909
    nmb_call = NormalizedMeanBias(forecast_type="0.5")
    nmb_instance = nmb_call()
    nmb_instance.update(data)
    result = nmb_instance.get()
    assert pytest.approx(result) == 0.0909090909
    assert "NMB" in nmb_instance.name


def test_signed_deviation_stats():
    data = {"label": np.array([[105.0, 95.0]]), "0.5": np.array([[100.0, 100.0]])}
    s_err, t_scale = signed_deviation_stats(data, forecast_type="0.5")
    assert np.allclose(s_err, [[5.0, -5.0]])
    assert np.allclose(t_scale, [[105.0, 95.0]])


def test_metrics_with_list_inputs():
    # GiftEval/GluonTS sometimes provides lists instead of numpy arrays.
    # This test ensures we handle them via np.asanyarray conversion.
    data = {
        "input": [[10.0, 20.0], [30.0, 40.0]],  # List of lists
        "label": [[25.0], [45.0]],
        "0.5": [[22.0], [42.0]],
    }

    # 1. Test helper
    m_err, p_err = persistence_relative_error_stats(data, forecast_type="0.5")
    assert isinstance(m_err, np.ma.MaskedArray)
    assert np.allclose(m_err, [[3.0], [3.0]])

    # 2. Test full metric
    psr_call = PersistenceSkillRatio(forecast_type="0.5")
    psr_instance = psr_call()
    psr_instance.update(data)
    result = psr_instance.get()
    # Model error: 3.0, Persistence error: |25-20|=5, |45-40|=5. Mean PErr = 5.
    # PSR = 3 / 5 = 0.6
    assert pytest.approx(result) == 0.6
