"""
Defining some new metrics to measure biases and characteristics of timeseries forecasts

| Metric | Category | Ideal Value | What it detects |
| :--- | :--- | :--- | :--- |
| Persistence Skill Ratio (PSR) | Skill (Accuracy) | < 1.0 | Is the model better than a "naive" flat line starting from the last known value? |
| Mean Skill Ratio (MSR) | Skill (Accuracy) | < 1.0 | Is the model better than just guessing the historical average of the series? |
| Mean Attraction Ratio (MAR) | Dynamics (Energy) | 1.0 | Does the forecast match the global amplitude (distance from the mean) of the truth? |
| Path Volatility Ratio (PVR) | Dynamics (Energy) | 1.0 | Does the path have the correct "wiggliness" and transition from history? |
| Mean Bias Ratio (MBR) | Dynamics (Frequency) | 0.5 | How often does the model play it safe by landing between the mean and the target? |
| Normalized Mean Bias (NMB) | Direction | 0.0 | Is the model systematically "Bearish" (under-predicting) or "Bullish" (over-predicting)? |
| Relative Uncertainty (RU) | Sharpness | ~0.0* | How much "doubt" (interval width) does the model express relative to signal scale? |

*Ideal RU depends on data noise; in deterministic signals, it should be near 0.

Possible model behaviors:
- The "Lazy" Model: High PSR (>1) and high MSR (>1). The model is literally worse than a flat line.
- The "Muted" Model: MAR > 1.2 and PVR > 1.2. The model gets the direction right but significantly under-sizes the moves.
- The "Hallucinator": PVR < 0.8. The model is adding "jitter" or noise that isn't present in the ground truth.
- The "Hedger": MBR > 0.7. The model is systematically "leaning" toward the mean to minimize MSE loss.
- The "Terrified" Model: High RU (> 0.2) on simple patterns. The model is inflating its uncertainty intervals.
"""

from functools import partial
from gluonts.ev.metrics import BaseMetricDefinition, DirectMetric
from typing import Optional, Dict, Tuple

# Add Self to typing on Python < 3.11
try:
    from typing import Self  # noqa: F401
except ImportError:
    from typing_extensions import Self
from gluonts.ev.aggregations import Mean
from dataclasses import dataclass
import numpy as np

def _get_input_means(
    data: Dict[str, np.ndarray], context_length: Optional[int] = None
) -> np.ndarray:
    inputs = data["input"]
    per_series_means = []
    for series in inputs:
        series_np = np.asarray(series)
        series_slice = (
            series_np if context_length is None else series_np[-context_length:]
        )
        if series_slice.ndim in (1, 2):
            per_series_means.append(
                np.nanmean(series_slice, axis=series_slice.ndim - 1, keepdims=True)[0]
            )
        else:
            raise ValueError(f"Unsupported input dimensionality ({series_slice.ndim})")

    first_mean = per_series_means[0]
    if np.isscalar(first_mean):
        return np.asarray(per_series_means, dtype=float)[:, None]
    return np.ma.masked_invalid(np.stack(per_series_means, axis=0).astype(float))


def mean_abs_deviation_label(
    data: Dict[str, np.ndarray], context_length: Optional[int] = None
) -> np.ndarray:
    input_means = _get_input_means(data, context_length=context_length)
    label = np.asanyarray(data["label"])
    return np.ma.masked_invalid(np.abs(label - input_means))


def mean_abs_deviation_forecast(
    data: Dict[str, np.ndarray],
    forecast_type: str,
    context_length: Optional[int] = None,
) -> np.ndarray:
    input_means = _get_input_means(data, context_length=context_length)
    forecast = np.asanyarray(data[forecast_type])
    return np.ma.masked_invalid(np.abs(forecast - input_means))


class MEAN_DEVIATION_LABEL(BaseMetricDefinition):
    def __call__(
        self, axis: Optional[int] = None, context_length: Optional[int] = None
    ) -> DirectMetric:
        return DirectMetric(
            name="mean_abs_deviation_label",
            stat=partial(mean_abs_deviation_label, context_length=context_length),
            aggregate=Mean(axis=axis),
        )


@dataclass
class MEAN_DEVIATION_FORECAST(BaseMetricDefinition):
    forecast_type: str = "0.5"

    def __call__(
        self, axis: Optional[int] = None, context_length: Optional[int] = None
    ) -> DirectMetric:
        return DirectMetric(
            name=f"mean_abs_deviation_forecast[{self.forecast_type}]",
            stat=partial(
                mean_abs_deviation_forecast,
                forecast_type=self.forecast_type,
                context_length=context_length,
            ),
            aggregate=Mean(axis=axis),
        )


@dataclass
class MeanAttractionRatio:
    forecast_type: str = "0.5"
    context_length: Optional[int] = None

    def __call__(self, axis: Optional[int] = None):
        return _MARInstance(
            axis=axis,
            forecast_type=self.forecast_type,
            context_length=self.context_length,
        )


class _MARInstance(BaseMetricDefinition):
    def __init__(
        self,
        axis: Optional[int] = None,
        forecast_type: str = "0.5",
        context_length: Optional[int] = None,
    ) -> None:
        self.deviation_label = MEAN_DEVIATION_LABEL()(
            axis=axis, context_length=context_length
        )
        self.deviation_forecast = MEAN_DEVIATION_FORECAST(forecast_type=forecast_type)(
            axis=axis, context_length=context_length
        )
        # Updated name
        self.name = f"MeanAttractionRatio[{forecast_type}][{context_length if context_length else ''}]"

    def update(self, data: Dict[str, np.ndarray]) -> Self:
        self.deviation_label.update(data)
        self.deviation_forecast.update(data)
        return self

    def get(self) -> np.ndarray:
        target_dev = self.deviation_label.get()
        pred_dev = self.deviation_forecast.get()
        return target_dev / (pred_dev + 1e-10)


# ====================================== PathVolatilityRatio ======================================

def path_absolute_difference_label(data, context_length=None):
    inputs, label = data["input"], np.asanyarray(data["label"])
    # Extract the last element from each input series
    last_inputs = np.array([series[-1] if len(series) > 0 else np.nan for series in inputs])
    # Concatenate the last inputs with the label
    full_path = np.concatenate([last_inputs[..., None], label], axis=-1)
    return np.ma.masked_invalid(np.abs(np.diff(full_path, axis=-1)))

def path_absolute_difference_forecast(data, forecast_type, context_length=None):
    inputs, forecast = data["input"], np.asanyarray(data[forecast_type])
    # Extract the last element from each input series
    last_inputs = np.array([series[-1] if len(series) > 0 else np.nan for series in inputs])
    # Concatenate the last inputs with the forecast
    full_path = np.concatenate([last_inputs[..., None], forecast], axis=-1)
    return np.ma.masked_invalid(np.abs(np.diff(full_path, axis=-1)))

class PATH_VOLATILITY_LABEL(BaseMetricDefinition):
    def __call__(self, axis=None, context_length=None):
        return DirectMetric(
            name="path_volatility_label",
            stat=partial(path_absolute_difference_label, context_length=context_length),
            aggregate=Mean(axis=axis),
        )

@dataclass
class PATH_VOLATILITY_FORECAST(BaseMetricDefinition):
    forecast_type: str = "0.5"
    def __call__(self, axis=None, context_length=None):
        return DirectMetric(
            name=f"path_volatility_forecast[{self.forecast_type}]",
            stat=partial(path_absolute_difference_forecast, forecast_type=self.forecast_type, context_length=context_length),
            aggregate=Mean(axis=axis),
        )

@dataclass
class PathVolatilityRatio:
    forecast_type: str = "0.5"
    context_length: Optional[int] = None
    def __call__(self, axis=None):
        return _PathVolatilityRatioInstance(axis=axis, forecast_type=self.forecast_type, context_length=self.context_length)

class _PathVolatilityRatioInstance(BaseMetricDefinition):
    def __init__(self, axis=None, forecast_type="0.5", context_length=None):
        self.path_label = PATH_VOLATILITY_LABEL()(axis=axis, context_length=context_length)
        self.path_forecast = PATH_VOLATILITY_FORECAST(forecast_type=forecast_type)(axis=axis, context_length=context_length)
        # Updated name
        self.name = f"PathVolatilityRatio[{forecast_type}][{context_length if context_length else ''}]"

    def update(self, data):
        self.path_label.update(data)
        self.path_forecast.update(data)
        return self

    def get(self):
        return self.path_label.get() / (self.path_forecast.get() + 1e-10)


# ====================================== MeanBiasRatio ======================================

def mean_bias(data, forecast_type, context_length=None):
    input_means = _get_input_means(data, context_length=context_length)
    label, forecast = np.asanyarray(data["label"]), np.asanyarray(data[forecast_type])
    is_biased = (np.sign(label - input_means) == np.sign(label - forecast)).astype(float)
    is_biased[np.isnan(label) | np.isnan(forecast)] = np.nan
    return is_biased

@dataclass
class MeanBiasRatio(BaseMetricDefinition):
    forecast_type: str = "0.5"
    context_length: Optional[int] = None

    def __call__(self, axis: Optional[int] = None) -> DirectMetric:
        return DirectMetric(
            # Updated name
            name=f"MeanBiasRatio[{self.forecast_type}][{self.context_length if self.context_length else ''}]",
            stat=partial(mean_bias, forecast_type=self.forecast_type, context_length=self.context_length),
            aggregate=Mean(axis=axis),
        )


# ====================================== PersistenceSkillRatio ======================================

@dataclass
class PersistenceSkillRatio(BaseMetricDefinition):
    forecast_type: str = "0.5"
    def __call__(self, axis: Optional[int] = None):
        return _PSRInstance(axis=axis, forecast_type=self.forecast_type)

class _PSRInstance(BaseMetricDefinition):
    def __init__(self, axis=None, forecast_type="0.5"):
        self.axis, self.forecast_type = axis, forecast_type
        self.model_errors, self.persistence_errors = [], []
        # Updated name
        self.name = f"PersistenceSkillRatio[{forecast_type}]"

    def update(self, data):
        inputs, label, forecast = data["input"], np.asanyarray(data["label"]), np.asanyarray(data[self.forecast_type])
        # Extract the last element from each input series
        last_inputs = np.array([series[-1] if len(series) > 0 else np.nan for series in inputs])
        self.model_errors.append(np.ma.masked_invalid(np.abs(label - forecast)))
        self.persistence_errors.append(np.ma.masked_invalid(np.abs(label - last_inputs[..., None])))
        return self

    def get(self):
        return np.mean(np.concatenate(self.model_errors)) / (np.mean(np.concatenate(self.persistence_errors)) + 1e-10)


# ====================================== MeanSkillRatio ======================================

@dataclass
class MeanSkillRatio:
    forecast_type: str = "0.5"
    context_length: Optional[int] = None
    def __call__(self, axis=None):
        return _MSRInstance(axis=axis, forecast_type=self.forecast_type, context_length=self.context_length)

class _MSRInstance(BaseMetricDefinition):
    def __init__(self, axis=None, forecast_type="0.5", context_length=None):
        self.axis, self.forecast_type, self.context_length = axis, forecast_type, context_length
        self.model_errors, self.mean_errors = [], []
        # Updated name
        self.name = f"MeanSkillRatio[{forecast_type}][{context_length if context_length else ''}]"

    def update(self, data):
        input_means = _get_input_means(data, context_length=self.context_length)
        label, forecast = data["label"], data[self.forecast_type]
        self.model_errors.append(np.ma.masked_invalid(np.abs(label - forecast)))
        self.mean_errors.append(np.ma.masked_invalid(np.abs(label - input_means)))
        return self

    def get(self):
        return np.mean(np.concatenate(self.model_errors)) / (np.mean(np.concatenate(self.mean_errors)) + 1e-10)


# ====================================== NormalizedMeanBias ======================================

@dataclass
class NormalizedMeanBias:
    forecast_type: str = "0.5"
    def __call__(self, axis=None):
        return _DirectionalBiasInstance(axis=axis, forecast_type=self.forecast_type)

class _DirectionalBiasInstance(BaseMetricDefinition):
    def __init__(self, axis=None, forecast_type="0.5"):
        self.axis, self.forecast_type = axis, forecast_type
        self.signed_errors, self.target_scales = [], []
        # Updated name
        self.name = f"NormalizedMeanBias[{forecast_type}]"

    def update(self, data):
        label, forecast = np.asanyarray(data["label"]), np.asanyarray(data[self.forecast_type])
        self.signed_errors.append(np.ma.masked_invalid(label - forecast))
        self.target_scales.append(np.ma.masked_invalid(np.abs(label)))
        return self

    def get(self):
        return np.mean(np.concatenate(self.signed_errors)) / (np.mean(np.concatenate(self.target_scales)) + 1e-10)


# ====================================== RelativeUncertainty ======================================

@dataclass
class RelativeUncertainty:
    upper_quantile: str = "0.9"
    lower_quantile: str = "0.1"

    def __call__(self, axis: Optional[int] = None):
        return _RelativeUncertaintyInstance(
            axis=axis, 
            upper=self.upper_quantile, 
            lower=self.lower_quantile
        )

class _RelativeUncertaintyInstance(BaseMetricDefinition):
    def __init__(self, axis: Optional[int], upper: str, lower: str):
        self.upper, self.lower = upper, lower
        # Updated name
        self.name = f"RelativeUncertainty[{self.lower}-{self.upper}]" 
        self.widths, self.scales = [], []

    def update(self, data: Dict[str, np.ndarray]) -> "Self":
        self.widths.append(np.ma.masked_invalid(data[self.upper] - data[self.lower]))
        self.scales.append(np.ma.masked_invalid(np.abs(data["label"])))
        return self

    def get(self) -> np.ndarray:
        return np.mean(np.concatenate(self.widths)) / (np.mean(np.concatenate(self.scales)) + 1e-10)