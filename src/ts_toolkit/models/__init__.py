import importlib

from .benchmarks import (
    ADIDA,
    IMAPA,
    AutoARIMA,
    AutoCES,
    AutoETS,
    CrostonClassic,
    DynamicOptimizedTheta,
    HistoricAverage,
    SeasonalNaive,
    Theta,
    ZeroModel,
)

MODEL_REGISTRY = {
    "Moirai": "ts_toolkit.models.foundational.moirai",
    "Toto": "ts_toolkit.models.foundational.toto",
    "Sundial": "ts_toolkit.models.foundational.sundial",
    "Chronos2": "ts_toolkit.models.foundational.chronos2",
    #TODO fill the rest
}

MULTIVARIATE_MODEL_LIST = ['toto', 'moirai', 'chronos2'] 

def get_model(model_name: str):
    module_path = MODEL_REGISTRY[model_name]
    module = importlib.import_module(module_path)
    return getattr(module, model_name)



__all__ = [
    "ADIDA",
    "IMAPA",
    "AutoARIMA",
    "AutoCES",
    "AutoETS",
    "CrostonClassic",
    "DynamicOptimizedTheta",
    "HistoricAverage",
    "SeasonalNaive",
    "Theta",
    "ZeroModel",
]
