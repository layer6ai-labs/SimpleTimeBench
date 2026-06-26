import yaml
from types import SimpleNamespace
import pandas as pd
from dotenv import load_dotenv
import os
import time
import pickle

load_dotenv()
GIFT_EVAL_PATH = os.getenv("GIFT_EVAL")
REPO_ROOT = os.getenv("REPO_ROOT")

# assert if both env vars are set
if GIFT_EVAL_PATH is None or REPO_ROOT is None:
    raise ValueError(
        "Environment variables GIFT_EVAL and REPO_ROOT must be set in the .env file."
    )


def load_config(config_path=None, config_dict=None, overrides_path=None) -> SimpleNamespace:
    """
    Loads configuration from a base YAML file and applies optional overrides.
    """
    if config_path:
        print(f"Loading base config from {REPO_ROOT}/{config_path}...")
        config_path = os.path.join(REPO_ROOT, config_path)
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    elif config_dict:
        config = config_dict
    else:
        raise ValueError("Either config_path or config_dict must be provided.")
    
    if overrides_path:
        print(f"Loading overrides from {overrides_path}...")
        with open(overrides_path, "r") as f:
            overrides = yaml.safe_load(f)

        def merge_dicts(dict1, dict2):
            for k, v in dict2.items():
                if k in dict1 and isinstance(dict1[k], dict) and isinstance(v, dict):
                    dict1[k] = merge_dicts(dict1[k], v)
                else:
                    dict1[k] = v
            return dict1

        config = merge_dicts(config, overrides)

    # Add REPO_ROOT to config paths
    if "paths" in config:
        for key, value in config["paths"].items():
            if isinstance(value, str) and not os.path.isabs(value):
                config["paths"][key] = os.path.join(REPO_ROOT, value)
        #
        if "output_path" in config["paths"]:
            experiment_name = config.get("experiment_name", "unnamed_experiment")
            # add time stamp to experiment name to avoid overwriting
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            experiment_name = f"{experiment_name}_{timestamp}"
            config["paths"]["output_path"] = os.path.join(
                config["paths"]["output_path"], experiment_name
            )
            config["experiment_name"] = experiment_name
            os.makedirs(config["paths"]["output_path"], exist_ok=True)

    # Convert dictionary to SimpleNamespace for dot access
    return SimpleNamespace(
        **{
            k: SimpleNamespace(**v) if isinstance(v, dict) else v
            for k, v in config.items()
        }
    )

def forecast_output_saver(pipeline, multivariate_str, model_name, output_file):
    output_list = pipeline.output_forecast_generator()

    output_data = {multivariate_str: {model_name: {}}}
    # write the outputs to a json/pickle file
    for output in output_list:
        input_list, label_list, forecast_list, data, item_id, dataset_label = output
        if dataset_label not in output_data[multivariate_str][model_name]:
            output_data[multivariate_str][model_name][dataset_label] = {}
        
        output_data[multivariate_str][model_name][dataset_label][item_id] = {
            "dataset_label": dataset_label,
            "input_list": input_list,
            "label_list": label_list,
            "forecast_list": forecast_list,
            "data": data,
            "item_id": item_id,
        }
                
    # append to existing file or create new
    if os.path.exists(output_file):
        with open(output_file, "rb") as f:
            loaded_output_data = pickle.load(f)
    else:
        loaded_output_data = {}

    # deep merge two dictionaries
    for mv_key in output_data.keys():
        if mv_key not in loaded_output_data:
            loaded_output_data[mv_key] = {}
        for model_key in output_data[mv_key].keys():
            if model_key not in loaded_output_data[mv_key]:
                loaded_output_data[mv_key][model_key] = {}
            for dataset_key in output_data[mv_key][model_key].keys():
                if dataset_key not in loaded_output_data[mv_key][model_key]:
                    loaded_output_data[mv_key][model_key][dataset_key] = {}
                for item_id_key, item_data in output_data[mv_key][model_key][dataset_key].items():
                    loaded_output_data[mv_key][model_key][dataset_key][item_id_key] = item_data
            
    with open(output_file, "wb") as f:
        pickle.dump(loaded_output_data, f)
    
    return loaded_output_data

# df counterpart of MultivariateToUnivariate from dataloaders/base_loader.py for dataframes
def multi2univariate_df(
    df: pd.DataFrame, target_fields: list[str], field: str = "y"
) -> pd.DataFrame:
    univariate_dfs = []
    # for item_id, group in df.groupby("item_id"):
    for dim, col in enumerate(target_fields):
        univariate_entry = df[["unique_id", "ds"]].copy()
        univariate_entry[field] = df[col]
        univariate_entry["unique_id"] = df["unique_id"] + "_dim" + str(dim)
        univariate_dfs.append(univariate_entry)
    return pd.concat(univariate_dfs, ignore_index=True)


def uni2multvariate_df(
    df: pd.DataFrame, target_fields: list[str], field: str = "y"
) -> pd.DataFrame:
    multivariate_df = df[["unique_id", "ds"]].copy()
    for dim in range(len(target_fields)):
        multivariate_df[target_fields[dim]] = df[
            df["unique_id"].str.endswith(f"_dim{dim}")
        ][field].values
    return multivariate_df
