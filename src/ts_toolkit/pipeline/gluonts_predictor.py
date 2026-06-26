from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm
import utilsforecast.processing as ufp
from gluonts.dataset import Dataset
from gluonts.dataset.util import forecast_start
from gluonts.model import Forecast
from gluonts.model.forecast import QuantileForecast
from gluonts.model.predictor import RepresentablePredictor
from gluonts.transform.feature import LastValueImputation, MissingValueImputation
from ..models.utils.forecaster import Forecaster
from ..dataloaders.utils import QUANTILE_LEVELS
from collections import OrderedDict

import logging

logger = logging.getLogger(__name__)

MAX_DATASET_SIZE = float("inf")


class GluonTSPredictor(RepresentablePredictor):
    """
    Adapter to use a TimeCopilot Forecaster as a GluonTS Predictor.

    This class wraps a TimeCopilot Forecaster and exposes the GluonTS Predictor
    interface, allowing it to be used with GluonTS evaluation and processing utilities.
    """

    def __init__(
        self,
        forecaster: Forecaster,
        h: int | None = None,
        freq: str | None = None,
        level: list[int | float] | None = None,
        quantiles: list[float] | None = None,
        max_length: int | None = None,
        imputation_method: MissingValueImputation | None = None,
        batch_size: int | None = 1024,
    ):
        """
        Initialize a GluonTSPredictor.

        Args:
            forecaster (Forecaster): The TimeCopilot forecaster to wrap.
                You can use any forecaster from TimeCopilot, and create your own
                forecaster by subclassing
                [Forecaster][timecopilot.models.utils.forecaster.Forecaster].
            h (int | None): Forecast horizon. If None (default), the horizon is
                inferred from the dataset.
            freq (str | None): Frequency string (e.g., 'D', 'H').
                If None (default), the frequency is inferred from the dataset.
            level (list[int | float] | None): Not supported; use quantiles instead.
            quantiles (list[float] | None): Quantiles to forecast. If None (default),
                the default quantiles [0.1, 0.2, ..., 0.9] are used.
            max_length (int | None): Maximum length of input series.
            imputation_method (MissingValueImputation | None): Imputation method for
                missing values. If None (default), the last value is used
                with LastValueImputation().
            batch_size (int | None): Batch size for prediction.

        Raises:
            NotImplementedError: If level is provided (use quantiles instead).
        """
        self.forecaster = forecaster
        self.h = h
        self.freq = freq
        self.level = level
        if level is not None:
            raise NotImplementedError("level is not supported, use quantiles instead")
        self.quantiles = quantiles or QUANTILE_LEVELS
        self.max_length = max_length
        self.imputation_method = imputation_method or LastValueImputation()
        self.batch_size = batch_size
        self.alias = forecaster.alias

    def _gluonts_dataset_to_df(
        self,
        dataset: Dataset,
    ) -> pd.DataFrame:  # tuple[pd.DataFrame, dict[str, Any]]:

        dfs: list[pd.DataFrame] = []
        self.univar_metadata: dict[str, Any] = {}
        self.multivar_metadata: dict[str, Any] = {}

        for _, entry in enumerate(dataset):
            target = np.asarray(entry["target"], dtype=np.float32)
            if target.ndim == 1:
                target = target[np.newaxis, :]
            # swap the axes to have shape (n_timesteps, n_variates)
            target = target.T  # entry["target"]'s shape is (n_variates, n_timesteps)

            if self.max_length is not None and len(target) > self.max_length:
                entry["start"] += len(target[: -self.max_length])
                target = target[-self.max_length :]  # noqa: E203
                entry["target"] = target.T

            if np.isnan(target).any():
                target = np.apply_along_axis(self.imputation_method, axis=0, arr=target)

            fcst_start = forecast_start(entry)

            uid = f"{entry['item_id']}-{fcst_start}"
            ds_start = entry["start"]
            ds = pd.date_range(
                start=ds_start.to_timestamp(),
                freq=ds_start.freq,
                periods=len(target),
            )
            targets_dict = (
                {f"y_{i}": target[:, i] for i in range(target.shape[1])}
                if target.shape[1] > 1
                else {"y": target.squeeze()}
            )
            self.target_cols = list(targets_dict.keys())

            uid_df = pd.DataFrame(
                {
                    "unique_id": [uid] * target.shape[0],
                    "ds": ds,
                    **targets_dict,
                }
            )
            dfs.append(uid_df)
            self.univar_metadata[uid] = {
                "item_id": entry["item_id"],
                "fcst_start": fcst_start,
            }
            for dim in range(len(self.target_cols)):
                self.multivar_metadata[f"{uid}_dim{dim}"] = {
                    "item_id": f"{entry['item_id']}_dim{dim}",
                    "fcst_start": fcst_start,
                }

        self.reoorder_metadata()
        df = pd.concat(dfs, ignore_index=True)
        return df

    def reoorder_metadata(self):
        if len(self.target_cols) == 1:
            return  # No need to reorder for univariate case
        # Extract the keys and group them by dimension
        grouped = {}
        for key in self.multivar_metadata.keys():
            item, dim = key.split("_dim")
            grouped.setdefault(dim, []).append(key)

        # Flatten and reorder
        flattened = OrderedDict()
        for dim in sorted(grouped):  # Sort dimensions
            for key in grouped[dim]:  # Add items in order
                flattened[key] = self.multivar_metadata[key]

        # Result
        self.multivar_metadata = {
            k: self.multivar_metadata[k] for k in list(flattened.keys())
        }

    def _predict_df(
        self,
        df: pd.DataFrame,
        metadata: dict[str, Any],
        h: int,
        freq: str,
    ) -> list[Forecast]:
        logger.debug(f"\n\nForecaster input df shape: {df.shape}, h: {h}, freq: {freq}")

        fcst_df = self.forecaster.forecast(
            df=df,
            h=h,
            freq=freq,
            level=self.level,
            quantiles=self.quantiles,
        )

        fcst_df = fcst_df.set_index("unique_id")
        fcsts: list[Forecast] = []
        for uid, metadata_uid in tqdm(
            metadata.items(), total=len(metadata), disable=True
        ):
            fcst_df_uid = fcst_df.loc[uid]
            forecast_arrays = ufp.value_cols_to_numpy(
                df=fcst_df_uid,
                id_col="unique_id",
                time_col="ds",
                target_col=None,
            )
            forecast_keys = ["mean"]
            if self.quantiles is not None:
                forecast_keys += [f"{q}" for q in self.quantiles]
            q_fcst = QuantileForecast(
                forecast_arrays=forecast_arrays.T,
                forecast_keys=forecast_keys,
                item_id=metadata_uid["item_id"],
                start_date=metadata_uid["fcst_start"],
            )
            fcsts.append(q_fcst)
        return fcsts

    def _predict_batch(
        self,
        batch: list[Dataset],
        h: int,
        freq: str,
    ) -> list[Forecast]:
        df = self._gluonts_dataset_to_df(batch)
        # TODO configurable evaluate all variates or just one
        metadata = (
            self.multivar_metadata
            if len(self.target_cols) > 1
            else self.univar_metadata
        )
        logger.debug(f"Batch size: {len(batch)}, metadata size: {len(metadata)}, df: {df.shape}")
        logger.debug(df)
        # temporarily disable predicting large datasets
        if len(df) > MAX_DATASET_SIZE:
            raise ValueError(
                f"Dataset is of size {len(df)}. Predicting datasets larger than {MAX_DATASET_SIZE} is disabled for now."
            )
        return self._predict_df(df=df, metadata=metadata, h=h, freq=freq)

    def predict(self, dataset: Dataset, **kwargs: Any) -> list[Forecast]:
        """
        Predict forecasts for a GluonTS Dataset.

        Args:
            dataset (Dataset): GluonTS Dataset to forecast.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            list[Forecast]: List of GluonTS Forecast objects for the dataset.
        """
        fcsts: list[Forecast] = []
        batch: list[Dataset] = []
        h = self.h or dataset.test_data.prediction_length
        if h is None:
            raise ValueError("horizon `h` must be provided")
        freq = self.freq
        for _, entry in tqdm(enumerate(dataset), total=len(dataset), disable=True):
            if freq is None:
                freq = entry["freq"]
            batch.append(entry)
            if len(batch) == self.batch_size:
                fcsts.extend(self._predict_batch(batch=batch, h=h, freq=freq))
                batch = []
        if len(batch) > 0:
            if freq is None:
                raise ValueError("frequency `freq` must be provided")
            fcsts.extend(self._predict_batch(batch=batch, h=h, freq=freq))
        return fcsts
