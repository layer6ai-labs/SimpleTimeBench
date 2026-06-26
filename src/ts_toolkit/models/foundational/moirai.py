from contextlib import contextmanager

import torch
from gluonts.torch.model.predictor import PyTorchPredictor
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
from uni2ts.model.moirai_moe import MoiraiMoEForecast, MoiraiMoEModule

from ..utils.gluonts_forecaster import GluonTSForecaster


class Moirai(GluonTSForecaster):
    """
    Moirai is a universal foundation model for time series forecasting, designed to
    handle a wide range of frequencies, multivariate series, and covariates. It uses
    a masked encoder-based transformer architecture with multi-patch size projection
    layers and Any-Variate Attention, enabling zero-shot and probabilistic
    forecasting. See the [official repo](https://github.com/
    SalesforceAIResearch/uni2ts),
    [Hugging Face](https://huggingface.co/collections/
    Salesforce/moirai-r-models-65c8d3a94c51428c300e0742), and
    [arXiv:2402.02592](https://arxiv.org/abs/2402.02592) for more details.
    """

    def __init__(
        self,
        repo_id: str = "Salesforce/moirai-1.0-R-large",
        filename: str = "model.ckpt",
        context_length: int = 4096,
        patch_size: int = 32,
        num_samples: int = 100,
        target_dim: int = 1,
        feat_dynamic_real_dim: int = 0,
        past_feat_dynamic_real_dim: int = 0,
        batch_size: int = 32,
        alias: str = "Moirai",
    ):
        """
        Args:
            repo_id (str, optional): The Hugging Face Hub model ID or local path to
                load the Moirai model from. Examples include
                "Salesforce/moirai-1.0-R-large". Defaults to
                "Salesforce/moirai-1.0-R-large". See the full list of models at
                [Hugging Face](https://huggingface.co/collections/Salesforce/
                moirai-r-models-65c8d3a94c51428c300e0742).
            filename (str, optional): Checkpoint filename for the model weights.
                Defaults to "model.ckpt".
            context_length (int, optional): Maximum context length (input window size)
                for the model. Controls how much history is used for each forecast.
                Defaults to 4096.
            patch_size (int, optional): Patch size for patch-based input encoding.
                Can be set to "auto" or a specific value (e.g., 8, 16, 32, 64, 128).
                Defaults to 32. See the Moirai paper for recommended values by
                frequency. Not used for Moirai-2.0.
            num_samples (int, optional): Number of samples for probabilistic
                forecasting. Controls the number of forecast samples drawn for
                uncertainty estimation. Defaults to 100.
                Not used for Moirai-2.0.
            target_dim (int, optional): Number of target variables (for multivariate
                forecasting). Defaults to 1.
            feat_dynamic_real_dim (int, optional): Number of dynamic real covariates
                known in the future. Defaults to 0.
            past_feat_dynamic_real_dim (int, optional): Number of past dynamic real
                covariates. Defaults to 0.
            batch_size (int, optional): Batch size to use for inference. Defaults to
                32. Adjust based on available memory and model size.
            alias (str, optional): Name to use for the model in output DataFrames and
                logs. Defaults to "Moirai".

        Notes:
            **Academic Reference:**

            - Paper: [Unified Training of Universal Time Series Forecasting Transformers](https://arxiv.org/abs/2402.02592)

            **Resources:**

            - GitHub: [SalesforceAIResearch/uni2ts](https://github.com/SalesforceAIResearch/uni2ts)
            - HuggingFace: [Salesforce/moirai-r-models](https://huggingface.co/collections/Salesforce/moirai-r-models-65c8d3a94c51428c300e0742)

            **Technical Details:**

            - The model is loaded onto the best available device (GPU if available,
              otherwise CPU).
        """
        super().__init__(
            repo_id=repo_id,
            filename=filename,
            alias=alias,
            num_samples=num_samples,
        )
        self.context_length = context_length
        self.patch_size = patch_size
        self.target_dim = target_dim
        self.feat_dynamic_real_dim = feat_dynamic_real_dim
        self.past_feat_dynamic_real_dim = past_feat_dynamic_real_dim
        self.batch_size = batch_size

    @contextmanager
    def get_predictor(self, prediction_length: int) -> PyTorchPredictor:
        kwargs = {
            "prediction_length": prediction_length,
            "context_length": self.context_length,
            "patch_size": self.patch_size,
            "num_samples": self.num_samples,
            "target_dim": self.target_dim,
            "feat_dynamic_real_dim": self.feat_dynamic_real_dim,
            "past_feat_dynamic_real_dim": self.past_feat_dynamic_real_dim,
        }
        if "moe" in self.repo_id:
            model_cls, model_module = MoiraiMoEForecast, MoiraiMoEModule
        elif "moirai-2.0" in self.repo_id:
            model_cls, model_module = Moirai2Forecast, Moirai2Module
            del kwargs["patch_size"]
            del kwargs["num_samples"]
        else:
            model_cls, model_module = MoiraiForecast, MoiraiModule
        model = model_cls(
            module=model_module.from_pretrained(self.repo_id),
            **kwargs,
        )
        predictor = model.create_predictor(batch_size=self.batch_size)

        try:
            yield predictor
        finally:
            del predictor, model
            torch.cuda.empty_cache()
