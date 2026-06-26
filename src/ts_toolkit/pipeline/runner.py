import numpy as np
import pandas as pd
import logging
from tqdm.notebook import tqdm
from ts_toolkit.models.ensembles.median import MedianEnsemble
from ts_toolkit.models import MULTIVARIATE_MODEL_LIST
from ts_toolkit.pipeline.gluonts_predictor import GluonTSPredictor
from ts_toolkit.pipeline.eval import Evaluator
from ts_toolkit.models import MODEL_REGISTRY, get_model
import os
from itertools import islice

logger = logging.getLogger(__name__)
class TsPipeline:
    """
    Orchestrates the evaluation of a forecaster on GIFT-Eval datasets.
    """

    def __init__(self, config):
        """
        Initializes the runner with configuration.
        Args:
            config: Configuration object with paths and model parameters.
        """

        self.config = config
        # max_length controls the df construction in src/ts_toolkit/pipeline/gluonts_predictor._gluonts_dataset_to_df()
        # which is the input data to the forecaster for prediction.
        # We are not sure if the context_length param of each model is effective in the MedianEnsemble forecaster,
        # So adding this control here to ensure the input context window is manageable.
        if not config.forecaster.max_length:
            config.forecaster.max_length = 512  # Default max_length if not specified
        self.output_path = config.paths.output_path if hasattr(config.paths, "output_path") else None
        self.storage_path = config.paths.storage_path
        self.forecaster_params = config.forecaster
        # --- Initialize Forecaster ---
        self.forecaster = self.instantiate_forecaster()
        # A GluonTS-compatible wrapper for any Forecaster from the toolkit.
        self.predictor = GluonTSPredictor(
            forecaster=self.forecaster,
            max_length=config.forecaster.max_length,
            batch_size=1_024,
        )
        self.populate_datasets()

        logger.info(
            f"   - Results will be saved to: {self.output_path}/{self.forecaster.alias}"
        )
        logger.info(f"   - Datasets will be loaded from: {self.storage_path}\n")

    # --- Helper function to instantiate forecaster ---
    def instantiate_forecaster(self):
        """instantiates the forecaster based on configuration."""
        forecaster_name = self.forecaster_params.name
        alias = self.forecaster_params.alias
        model_configs = self.forecaster_params.models

        models = []
        for model_cfg in model_configs:
            model_name = model_cfg.get("name")
            if model_name in MODEL_REGISTRY:
                model_class = get_model(model_name)
                models.append(model_class(**model_cfg.get("params", {})))
            else:
                logger.warning(
                    f"Unknown model type in config: {model_name}. Skipping."
                )

        if forecaster_name == "MedianEnsemble":
            return MedianEnsemble(models=models, alias=alias)
        else:
            raise ValueError(f"Unknown forecaster type: {forecaster_name}")

    def run(self, skip_existing: bool = True, num_plots: int = 5):
        """
        Evaluate the forecaster on the entire dataset.

        Args:
            dataset_name: The name of the dataset (e.g., 'm4_weekly').
            term: The evaluation term ('short', 'medium', 'long').
        """

        to_univariate = self.config.evaluation.to_univariate
        # TODO make univariate model-specific
        if (
            any(
                [
                    model["name"].lower() not in MULTIVARIATE_MODEL_LIST
                    for model in self.forecaster_params.models
                ]
            )
            and not to_univariate
        ):
            to_univariate = True
            logger.warning(
                f"Some models do not support multivariate series. to_univariate is set to {to_univariate} or use only multivariate models."  # noqa: E501
            )

        sel_visual_indices = range(len(self.config.datasets))
        # np.random.choice(
        #     np.arange(len(self.config.datasets)), 
        #     size=min(num_plots, len(self.config.datasets)), 
        #     replace=False
        # )

        # --- Run the evaluation for each configured dataset ---
        logger.info("\n--- Starting Evaluation ---")
        self.evaluators = []
        for idx, dataset_cfg in tqdm(enumerate(self.config.datasets), desc="Evaluating Datasets", total=len(self.config.datasets)):
            logger.debug(f"Evaluating dataset config: {dataset_cfg}")
            # Evaluator handles dataset loading and metric calculation.
            self.evaluator = Evaluator(
                dataset_name=dataset_cfg.get("name"),
                term=dataset_cfg.get("term", "short"),
                to_univariate=to_univariate,
                # Create a subdirectory for each model's results
                output_path=f"{self.output_path}/{self.forecaster.alias}",
                storage_path=self.storage_path,
                config=self.config,
            )
            try:
                # Update Moirai target_dim and re-instantiate forecaster if needed
                if self.evaluator.target_dim > 1 and not to_univariate:
                    for model_cfg in self.forecaster_params.models:
                        if model_cfg.get("name")== "Moirai":
                            model_cfg["params"]["target_dim"] = self.evaluator.target_dim
                            logger.info(self.forecaster_params.models)

                            self.forecaster = self.instantiate_forecaster()
                            self.predictor = GluonTSPredictor(
                                forecaster=self.forecaster,
                                max_length=self.config.forecaster.max_length,
                                batch_size=1_024,
                            )
                            logger.info("updated Moirai target dim with dataset target dim")

                # Run evaluation using GIFT-Eval's standardized metrics.
                self.evaluator.evaluate_predictor(self.predictor, batch_size=512)
                self.results_df = self.evaluator.get_results()
                if self.config.evaluation.show_plot and idx in sel_visual_indices:
                    self.evaluator.plot_forecasts(
                        num_plots=num_plots, show_plot=self.config.evaluation.show_plot
                    )
                self.evaluators.append(self.evaluator)
            except Exception as e:
                if skip_existing:
                    logger.error(
                        f"❌ ERROR evaluating {dataset_cfg.get('name')} ({dataset_cfg.get('term')}): {e}"
                    )
                    logger.error("Skipping to the next dataset...")
                    # Continue to the next task even if one fails
                    continue
                else:
                    raise e

        logger.info("\n--- ✅ All evaluations complete. ---")

    def populate_datasets(self):
        """Populates self.config.datasets if not already specified in the config file."""
        if not hasattr(self.config, "datasets") or not self.config.datasets:
            logger.info(
                f"No datasets specified in config.datasets. Discovering datasets from: {self.storage_path}"
            )
            discovered_datasets = []
            if os.path.exists(self.storage_path) and os.path.isdir(self.storage_path):
                for folder in os.listdir(self.storage_path):
                    folder_path = os.path.join(self.storage_path, folder)
                    if os.path.isdir(folder_path):
                        for subfolder in os.listdir(folder_path):
                            subfolder_path = os.path.join(folder_path, subfolder)
                            if os.path.isdir(subfolder_path):
                                # Assuming the subfolder name can be used as the dataset name
                                discovered_datasets.append(
                                    {"name": f"{folder}/{subfolder}"}
                                )
                self.config.datasets = discovered_datasets
                logger.info(
                    f"Discovered datasets: {[d['name'] for d in discovered_datasets]}"
                )
            else:
                logger.info(
                    f"Warning: storage_path '{self.storage_path}' does not exist or is not a directory."
                )
                self.config.datasets = (
                    []
                )  # Ensure it's an empty list if path is invalid

    def get_results(self) -> pd.DataFrame:
        return self.results_df
    
    def get_forecast_horizons(self) -> dict:
        horizons = {}
        for dataset_cfg in tqdm(self.config.datasets):
            # Evaluator handles dataset loading and metric calculation.
            evaluator = Evaluator(
                dataset_name=dataset_cfg.get("name"),
                term=dataset_cfg.get("term", "short"),
                to_univariate=False,
                # Create a subdirectory for each model's results
                output_path=f"{self.output_path}/{self.forecaster.alias}",
                storage_path=self.storage_path,
                config=self.config,
            )
            print(f"Dataset: {dataset_cfg.get('name')}, Horizon: {evaluator.dataset.test_data.prediction_length}")
            horizons[dataset_cfg.get("name")] = evaluator.dataset.test_data.prediction_length
        return horizons
    
    def output_forecast_generator(self, num_outputs: int | None = None):
        self.output_data = []
        
        for evaluator in self.evaluators:
            dataset = evaluator.dataset.test_data
            num_generated = 0            
            for data, next_data in zip(dataset, islice(dataset, 1, None)):
                assert data[0]['item_id'] == data[1]['item_id'], f"Data input item_id {data[0]['item_id']} does not match label item_id {data[1]['item_id']}"
                item_id = data[0]['item_id'].split('_dim')[0]
                if evaluator.to_univariate or evaluator.target_dim == 1:
                    try:
                        if 'dim0' in data[0]['item_id'] or 'dim' not in data[0]['item_id']:
                            input_list, label_list = [data[0]['target']], [data[1]['target']]
                        else: # append next variate
                            input_list.append(data[0]['target'])
                            label_list.append(data[1]['target'])
                        # Check if the next data exists before accessing it
                        if next_data is not None and 'dim' in data[0]['item_id'] and 'dim0' not in next_data[0]['item_id']:
                            continue
                    except Exception as e:
                        print("data[0]['item_id'] = ", data[0]['item_id'])
                        print("data[0] = ", data[0])
                        raise e
                else:
                    input_list, label_list = data[0]["target"], data[1]["target"] # shape (num_variates, seq_len)
                
                # only plot num_plots items
                num_generated += 1
                if num_outputs is not None and num_generated > num_outputs:
                    break
                # aggregate all forecast variates for the current item_id
                forecast_list = []
                for fcst in evaluator.forecasts:
                    if fcst.item_id.split('_dim')[0] == item_id:
                        forecast_list.append(fcst)
                        
                self.output_data.append((
                        input_list, label_list, forecast_list, data, item_id, evaluator.dataset_name
                ))
                                   
        return self.output_data