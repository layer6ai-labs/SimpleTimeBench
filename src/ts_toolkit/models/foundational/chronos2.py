from contextlib import contextmanager
import logging

import numpy as np
import pandas as pd
import torch
from chronos import Chronos2Pipeline, BaseChronosPipeline
from tqdm import tqdm

from ..utils.forecaster import Forecaster, QuantileConverter
from .utils import TimeSeriesDataset

logger = logging.getLogger(__name__)


class Chronos2(Forecaster):
    """
    Chronos2 is a family of open foundation models for time series forecasting.
    It supports covariates for enhanced forecasting.
    See the [official repo](https://github.com/autogluon/chronos) for more details.
    """

    def __init__(
        self,
        repo_id: str = "s3://autogluon/chronos-2/", # Updated default to a smaller Chronos2 model
        prediction_length: int = 24,
        batch_size: int = 16,
        alias: str = "Chronos2",
    ):
        """
        Args:
            repo_id (str, optional): The Hugging Face Hub model ID or local path to
                load the Chronos2 model from. Examples include "s3://autogluon/chronos-2/".
                Defaults to "s3://autogluon/chronos-2/". See the full list of models
                at [Hugging Face](https://huggingface.co/autogluon).
            prediction_length (int, optional): Number of steps to forecast. This
                will be passed directly to the `predict_df` method. Defaults to 24.
            batch_size (int, optional): Batch size to use for inference. Defaults to 16.
                Adjust based on available memory and model size.
            alias (str, optional): Name to use for the model in output DataFrames and
                logs. Defaults to "Chronos2".

        Notes:
            **Academic Reference:**

            - Paper: [Chronos: Learning the Language of Time Series](https://arxiv.org/abs/2403.02102)

            **Resources:**

            - GitHub: [autogluon/chronos](https://github.com/autogluon/chronos)
            - HuggingFace: [autogluon Models](https://huggingface.co/autogluon)

            **Technical Details:**

            - The model is loaded onto the best available device (GPU if available,
              otherwise CPU).
            - For best performance, a CUDA-capable GPU is recommended.
        """
        self.repo_id = repo_id
        self.prediction_length = prediction_length
        self.batch_size = batch_size
        self.alias = alias
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @contextmanager
    def _get_model(self) -> Chronos2Pipeline:
        """Loads the Chronos2 pipeline."""
        pipeline = BaseChronosPipeline.from_pretrained(self.repo_id, device_map=self.device)
        try:
            yield pipeline
        finally:
            # Clean up, though Chronos2Pipeline might handle its own memory
            del pipeline
            if self.device == "cuda":
                torch.cuda.empty_cache()

    def forecast(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str | None = None,
        level: list[int | float] | None = None,
        quantiles: list[float] | None = None,
        id_column: str = "unique_id",
        timestamp_column: str = "ds",
    ) -> pd.DataFrame:
        """Generate forecasts for time series data using the Chronos2 model.

        This method produces point forecasts and, optionally, prediction
        intervals or quantile forecasts. The input DataFrame can contain one
        or multiple time series in stacked (long) format. Chronos2 also accepts
        covariates.

        Args:
            df (pd.DataFrame):
                DataFrame containing the time series to forecast. It must
                include as columns:

                    - "unique_id": an ID column to distinguish multiple series.
                    - "ds": a time column indicating timestamps or periods.
                    - "y": a target column with the observed values.
                    - Optional covariate columns as specified in `covariate_columns`.

            h (int):
                Forecast horizon specifying how many future steps to predict.
                This value will override `self.prediction_length` if provided.
            freq (str, optional):
                Frequency of the time series (e.g. "D" for daily, "M" for
                monthly). See [Pandas frequency aliases](https://pandas.pydata.org/
                pandas-docs/stable/user_guide/timeseries.html#offset-aliases) for
                valid values. If not provided, the frequency will be inferred
                from the data.
            level (list[int | float], optional):
                Confidence levels for prediction intervals, expressed as
                percentages (e.g. [80, 95]). If provided, the returned
                DataFrame will include lower and upper interval columns for
                each specified level.
            quantiles (list[float], optional):
                List of quantiles to forecast, expressed as floats between 0
                and 1. Should not be used simultaneously with `level`. When
                provided, the output DataFrame will contain additional columns
                named in the format "model-q-{percentile}", where {percentile}
                = 100 × quantile value.
            id_column (str, optional): The name of the column that identifies
                individual time series. Defaults to "unique_id".
            timestamp_column (str, optional): The name of the column containing
                timestamp information. Defaults to "ds".
            target_column (str, optional): The name of the column containing
                the target variable to forecast. Defaults to "y".
            covariate_columns (list[str], optional): A list of column names that
                represent covariates. These columns must be present in `df`
                for historical values and will be used to construct `future_df`.

        Returns:
            pd.DataFrame:
                DataFrame containing forecast results. Includes:

                    - point forecasts for each timestamp and series.
                    - prediction intervals if `level` is specified.
                    - quantile forecasts if `quantiles` is specified.

                For multi-series data, the output retains the same unique
                identifiers as the input DataFrame.
        """
        
        # Ensure timestamp column is datetime
        df[timestamp_column] = pd.to_datetime(df[timestamp_column])
        
        freq = self._maybe_infer_freq(df, freq)
        qc = QuantileConverter(level=level, quantiles=quantiles)

        # Prepare context_df
        # Chronos expects target column to be named 'target' for prediction
        target_columns = [col for col in df.columns if col.startswith("y")]
        fcst_df_all = pd.DataFrame()
        for target_col in target_columns:
            dim = int(target_col.split("y_")[-1]) if target_col != "y" else None
            context_df = df.copy()
            covariate_columns = [col for col in df.columns if col not in [id_column, timestamp_column, target_col]]
            context_df = context_df.rename(columns={
                target_col: "target", id_column: "id", timestamp_column: "timestamp"
            })
            # Prepare future_df for covariates if provided
            future_df = None # TODO
             
            with self._get_model() as pipeline:
                pred_df = pipeline.predict_df(
                    context_df,
                    future_df=future_df,
                    prediction_length=h, # Use the h passed to the forecast method
                    quantile_levels=qc.quantiles,
                    id_column="id",
                    timestamp_column="timestamp",
                    target="target",
                    # Chronos2Pipeline automatically handles target column, no need to specify it explicitly here
                    # unless multiple target columns are supported, which isn't the case in the example.
                )

            # Rename columns to match the expected output format of the Forecaster base class
            # Chronos output columns: "id", "timestamp", "mean", "0.1", "0.5", "0.9" (for quantiles)
            
            # Convert quantile columns (e.g., '0.1', '0.5') to 'model-q-10', 'model-q-50' format
            renamed_cols = {
                "predictions": self.alias,
                "id": id_column,
                "timestamp": timestamp_column,
            }
            
            if qc.quantiles is not None:
                for q in qc.quantiles:
                    renamed_cols[str(q)] = f"{self.alias}-q-{int(q * 100)}"

            fcst_df = pred_df.rename(columns=renamed_cols)
            if dim is not None:
                fcst_df[id_column] = fcst_df[id_column].astype(str) + f"_dim{dim}" 

            # Remove the original target column 'target' from the output if it was mistakenly carried over
            if "target_name" in fcst_df.columns:
                fcst_df = fcst_df.drop(columns=["target_name"])

            # Apply quantile conversion to levels if specified
            if qc.level is not None and qc.quantiles is not None:
                fcst_df = qc.maybe_convert_quantiles_to_level(
                    fcst_df,
                    models=[self.alias],
                )
            fcst_df_all = pd.concat([fcst_df_all, fcst_df])
            
        return fcst_df_all