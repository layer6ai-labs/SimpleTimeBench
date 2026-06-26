from multivar_timeseries.evaluation.metrics import (
    MeanAttractionRatio,
    MeanBiasRatio,
    PersistenceSkillRatio,
    MeanSkillRatio,
    PathVolatilityRatio,
    NormalizedMeanBias,
    RelativeUncertainty,  # Added import
)
from gluonts.model import evaluate_forecasts
from gluonts.model.predictor import RepresentablePredictor
from gluonts.time_feature import get_seasonality
from huggingface_hub import snapshot_download
import yaml

from ..dataloaders import GiftEvalDatasetLoader
from .gluonts_predictor import GluonTSPredictor
from ..dataloaders.utils import QUANTILE_LEVELS
from .visualization import plot_forecast
import requests
from dotenv import load_dotenv
from gluonts.ev.metrics import (
    MAE,
    MAPE,
    MASE,
    MSE,
    MSIS,
    ND,
    NRMSE,
    RMSE,
    SMAPE,
    MeanWeightedSumQuantileLoss,
)
import json
import logging
from pathlib import Path
import pandas as pd
from types import SimpleNamespace

# Supress FutureWarning
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

logger = logging.getLogger(__name__)

load_dotenv()

# Updated Metric Suite
METRICS = [
    MeanAttractionRatio(),
    MeanBiasRatio(),
    PathVolatilityRatio(),
    PersistenceSkillRatio(),
    MeanSkillRatio(),
    NormalizedMeanBias(),
    RelativeUncertainty(),  # Added to suite
    MSE(forecast_type="mean"),
    MSE(forecast_type="0.5"),
    MAE(),
    MASE(),
    MAPE(),
    SMAPE(),
    MSIS(),
    RMSE(),
    NRMSE(),
    ND(),
    MeanWeightedSumQuantileLoss(quantile_levels=QUANTILE_LEVELS),
]

# DATASET_PROPERTIES_URL = "https://raw.githubusercontent.com/SalesforceAIResearch/gift-eval/refs/heads/main/notebooks/dataset_properties.json"
DATASET_PROPERTIES_URL = "https://raw.githubusercontent.com/SalesforceAIResearch/gift-eval/main/notebooks/dataset_properties.json"  # noqa: E501
REPO_ROOT = Path(__file__).resolve().parents[3]
LOCAL_DATASET_PROPERTIES_PATH = REPO_ROOT / "configs" / "dataset_properties.json"


class Evaluator:
    """
    Evaluation utility for [GIFTEval](https://huggingface.co/spaces/Salesforce/GIFT-Eval).

    This class loads a time series dataset, sets up evaluation metrics, and provides
    methods to evaluate GluonTS predictors on the dataset, saving results to CSV if
    desired.
    """

    @staticmethod
    def download_data(
        storage_path: Path | str | None = None, repo_id: str = "Salesforce/GiftEval"
    ):
        """
        Download the GIFTEval dataset from Hugging Face.
        """
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=storage_path,
        )

    def __init__(
        self,
        dataset_name: str,
        term: str,
        to_univariate: bool = False,
        output_path: Path | str | None = None,
        storage_path: Path | str | None = None,
        config: SimpleNamespace | None = None,
    ):
        # fmt: off
        """
        Initialize an instance for a specific dataset and evaluation term.

        Args:
            dataset_name (str): Name of the dataset to evaluate on.
            term (str): Evaluation term (e.g., 'medium', 'long').
            to_univariate (bool): Whether to convert multivariate series to univariate
            output_path (str | Path | None): Directory to save results CSV, or
                None to skip saving.
            storage_path (Path | str | None): Path where the dataset is stored.

        Raises:
            ValueError: If the dataset is not compatible with the specified term.

        """
        # fmt: on
        try:
            res_dataset_properties = requests.get(DATASET_PROPERTIES_URL, timeout=30)
            res_dataset_properties.raise_for_status()
            self.dataset_properties_map = res_dataset_properties.json()
        except requests.RequestException as exc:
            logger.warning(
                (
                    "Unable to reach %s (%s). Falling back to local dataset "
                    "properties at %s"
                ),
                DATASET_PROPERTIES_URL,
                exc,
                LOCAL_DATASET_PROPERTIES_PATH,
            )
            try:
                with LOCAL_DATASET_PROPERTIES_PATH.open(
                    "r", encoding="utf-8"
                ) as handle:
                    self.dataset_properties_map = json.load(handle)
            except (OSError, json.JSONDecodeError) as local_exc:
                raise RuntimeError(
                    "Failed to load dataset properties from remote and local sources."
                ) from local_exc

        self.target_dim = GiftEvalDatasetLoader(
            name=dataset_name, term=term, to_univariate=False, storage_path=storage_path
        ).target_dim
        updated_to_univariate = False if self.target_dim == 1 else to_univariate
        if updated_to_univariate != to_univariate:
            logger.info(
                "Dataset '%s' has target_dim=%d => to_univariate is forced set to %s.",
                dataset_name,
                self.target_dim,
                updated_to_univariate,
            )
        # Initialize the dataset
        self.dataset = GiftEvalDatasetLoader(
            name=dataset_name,
            term=term,
            to_univariate=updated_to_univariate,
            storage_path=storage_path,
        )
    
        logger.info(f"Input: {next(iter(self.dataset.test_data.input))['index'].max()}, label: {next(iter(self.dataset.test_data.label))['index'].max()}, prediction length: {self.dataset.prediction_length}")
        # logger.info(f"{next(iter(self.dataset.test_data.label))}")
        
        self.dataset_name = dataset_name
        self.term = term
        self.seasonality = get_seasonality(self.dataset.freq)
        self.output_path = output_path
        self.to_univariate = to_univariate
        self.results_df = pd.DataFrame()
        self.config = config
        self.eval_substr = config.evaluation.eval_substr if hasattr(config.evaluation, 'eval_substr') else None
        if config.forecaster.max_length:
            self.context_length = config.forecaster.max_length
        else:
            self.context_length = None

    def evaluate_predictor(
        self,
        predictor: RepresentablePredictor | GluonTSPredictor,
        batch_size: int | None = None,
        overwrite_results: bool = False,
    ) -> pd.DataFrame:
        """
        Evaluate a GluonTS predictor on the loaded dataset and save results.
        """
        if batch_size is None:
            if isinstance(predictor, GluonTSPredictor):
                batch_size = predictor.batch_size
            else:
                batch_size = 512

        self.forecasts = predictor.predict(self.dataset.test_data.input)
        
        eval_forecasts = self.forecasts
        test_data = self.dataset.univar_test_data
        if self.eval_substr:
            sub_forecasts = [
                fcst for fcst in self.forecasts if self.eval_substr in fcst.item_id
            ]
            if len(sub_forecasts) > 0:
                eval_forecasts = sub_forecasts
                test_data = self.dataset.univar_sub_test_data(self.eval_substr)
            else:
                logger.warning(
                    f"No forecasts found with eval_substr='{self.eval_substr}'. "
                    "Using the full dataset for evaluation."
                )
        
        res = evaluate_forecasts(
            forecasts=eval_forecasts,
            test_data=test_data,
            metrics=self.parse_metrics(METRICS),
            batch_size=batch_size,
            axis=None,
            mask_invalid_label=True,
            allow_nan_forecast=False,
            seasonality=self.seasonality,
        )

        self.model_name = (
            predictor.__class__.__name__
            if not isinstance(predictor, GluonTSPredictor)
            else predictor.alias
        )

        ctx_suffix = f"[{self.context_length}]" if self.context_length else ""

        results_data = [
            [
                f"{self.dataset_name}/{self.term}",
                "UNI" if self.to_univariate else "MULTI",
                self.target_dim,
                self.model_name,
                # --- Custom Metrics (Full names match self.name in metric classes) ---
                res.get(f"MeanAttractionRatio[0.5]{ctx_suffix}", [None])[0],
                res.get(f"MeanBiasRatio[0.5]{ctx_suffix}", [None])[0],
                res.get(f"PathVolatilityRatio[0.5]{ctx_suffix}", [None])[0],
                res.get("PersistenceSkillRatio[0.5]", [None])[0],
                res.get(f"MeanSkillRatio[0.5]{ctx_suffix}", [None])[0],
                res.get("NormalizedMeanBias[0.5]", [None])[0],
                res.get("RelativeUncertainty[0.1-0.9]", [None])[0],
                # --- Standard Metrics ---
                res.get("MSE[mean]", [None])[0],
                res.get("MSE[0.5]", [None])[0],
                res.get("MAE[0.5]", [None])[0],
                res.get("MASE[0.5]", [None])[0],
                res.get("MAPE[0.5]", [None])[0],
                res.get("sMAPE[0.5]", [None])[0],
                res.get("MSIS", [None])[0],
                res.get("RMSE[mean]", [None])[0],
                res.get("NRMSE[mean]", [None])[0],
                res.get("ND[0.5]", [None])[0],
                res.get("mean_weighted_sum_quantile_loss", [None])[0],
            ]
        ]

        # Create a DataFrame and write to CSV
        self.results_df = pd.DataFrame(
            results_data,
            columns=[
                "dataset",
                "variate",
                "num_variates",
                "model",
                # Custom Columns (Full names)
                f"eval_metrics/MeanAttractionRatio[0.5]{ctx_suffix}",
                f"eval_metrics/MeanBiasRatio[0.5]{ctx_suffix}",
                f"eval_metrics/PathVolatilityRatio[0.5]{ctx_suffix}",
                "eval_metrics/PersistenceSkillRatio[0.5]",
                f"eval_metrics/MeanSkillRatio[0.5]{ctx_suffix}",
                "eval_metrics/NormalizedMeanBias[0.5]",
                "eval_metrics/RelativeUncertainty[0.1-0.9]",
                # Standard Columns
                "eval_metrics/MSE[mean]",
                "eval_metrics/MSE[0.5]",
                "eval_metrics/MAE[0.5]",
                "eval_metrics/MASE[0.5]",
                "eval_metrics/MAPE[0.5]",
                "eval_metrics/sMAPE[0.5]",
                "eval_metrics/MSIS",
                "eval_metrics/RMSE[mean]",
                "eval_metrics/NRMSE[mean]",
                "eval_metrics/ND[0.5]",
                "eval_metrics/mean_weighted_sum_quantile_loss",
            ],
        )

        if self.output_path is not None:
            csv_file_path = Path(self.output_path) / "all_results.csv"
            csv_file_path.parent.mkdir(parents=True, exist_ok=True)
            if csv_file_path.exists() and not overwrite_results:
                self.results_df = pd.concat(
                    [pd.read_csv(csv_file_path), self.results_df]
                )
            self.results_df.to_csv(csv_file_path, index=False)

            logger.info(f"Results for {self.dataset_name} have been written to {csv_file_path}")
            config_yaml_path = (
                Path(self.output_path) / f"{self.config.experiment_name}_config.yaml"
            )
            with open(config_yaml_path, "w") as f:
                yaml.safe_dump(self.namespace_to_dict(self.config), f)
            logger.info(f"Config has been written to {config_yaml_path}")

    def namespace_to_dict(self, obj):
        if isinstance(obj, SimpleNamespace):
            return {k: self.namespace_to_dict(v) for k, v in vars(obj).items()}
        elif isinstance(obj, dict):
            return {k: self.namespace_to_dict(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.namespace_to_dict(v) for v in obj]
        else:
            return obj

    def get_results(self):
        return self.results_df

    def plot_forecasts(self, num_plots: int = 5, show_plot: bool = True):

        return plot_forecast(
            self.dataset.test_data,
            self.forecasts,
            self.to_univariate or (self.target_dim == 1),
            dataset_label=self.dataset_name,
            num_plots=num_plots,
            show_plot=show_plot,
        )

    def parse_metrics(self, METRICS):
        parsed_metrics = []
        for metric in METRICS:
            if hasattr(metric, "context_length"):
                metric.context_length = self.context_length
            parsed_metrics.append(metric)
        return parsed_metrics
