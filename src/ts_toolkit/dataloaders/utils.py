from itertools import product
from datasets import Dataset # To save in Hugging Face format in order to Match Gifteval
import pandas as pd
import os

QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

SHORT_DATASETS = [
    "m4_yearly",
    "m4_quarterly",
    "m4_monthly",
    "m4_weekly",
    "m4_daily",
    "m4_hourly",
    "electricity/15T",
    "electricity/H",
    "electricity/D",
    "electricity/W",
    "solar/10T",
    "solar/H",
    "solar/D",
    "solar/W",
    "hospital",
    "covid_deaths",
    "us_births/D",
    "us_births/M",
    "us_births/W",
    "saugeenday/D",
    "saugeenday/M",
    "saugeenday/W",
    "temperature_rain_with_missing",
    "kdd_cup_2018_with_missing/H",
    "kdd_cup_2018_with_missing/D",
    "car_parts_with_missing",
    "restaurant",
    "hierarchical_sales/D",
    "hierarchical_sales/W",
    "LOOP_SEATTLE/5T",
    "LOOP_SEATTLE/H",
    "LOOP_SEATTLE/D",
    "SZ_TAXI/15T",
    "SZ_TAXI/H",
    "M_DENSE/H",
    "M_DENSE/D",
    "ett1/15T",
    "ett1/H",
    "ett1/D",
    "ett1/W",
    "ett2/15T",
    "ett2/H",
    "ett2/D",
    "ett2/W",
    "jena_weather/10T",
    "jena_weather/H",
    "jena_weather/D",
    "bitbrains_fast_storage/5T",
    "bitbrains_fast_storage/H",
    "bitbrains_rnd/5T",
    "bitbrains_rnd/H",
    "bizitobs_application",
    "bizitobs_service",
    "bizitobs_l2c/5T",
    "bizitobs_l2c/H",
]

MED_LONG_DATASETS = [
    "electricity/15T",
    "electricity/H",
    "solar/10T",
    "solar/H",
    "kdd_cup_2018_with_missing/H",
    "LOOP_SEATTLE/5T",
    "LOOP_SEATTLE/H",
    "SZ_TAXI/15T",
    "M_DENSE/H",
    "ett1/15T",
    "ett1/H",
    "ett2/15T",
    "ett2/H",
    "jena_weather/10T",
    "jena_weather/H",
    "bitbrains_fast_storage/5T",
    "bitbrains_rnd/5T",
    "bizitobs_application",
    "bizitobs_service",
    "bizitobs_l2c/5T",
    "bizitobs_l2c/H",
]

ALL_DATASETS = SHORT_DATASETS + MED_LONG_DATASETS


DATASETS_WITH_TERMS = [(dataset_name, "short") for dataset_name in SHORT_DATASETS]
DATASETS_WITH_TERMS += [
    (dataset_name, term)
    for dataset_name, term in product(MED_LONG_DATASETS, ["medium", "long"])
]

def convert_csv_to_hf_dataset(data_dir, dataset_name, timestamp_col: str, frequency: str, date_format=None):
    """
    Convert a CSV file to a Hugging Face Dataset in the required format.

    Args:
        csv_path (str): Path to the CSV file.
        dataset_name (str): Name of the dataset to be saved.
        date_format (str, optional): Date format for parsing timestamps.
        timestamp_col (str, optional): Name of the timestamp column in the CSV.

    Returns:
        datasets.Dataset: The converted Hugging Face Dataset.
    """

    print(f"Loading CSV dataset from: {data_dir}")
    df = pd.read_csv(f"{data_dir}/{dataset_name}.csv")

    if date_format:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], format=date_format)
    else:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df.index = df[timestamp_col]
    df.drop(columns=[timestamp_col], inplace=True)
    
    # fill in missing dates with NaNs
    full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq=frequency)
    # 5. Reindex the dataframe
    df = df.reindex(full_range)
    df = df.ffill() # forward fill to fill in missing values, can be changed to other methods if needed
    # rename columns
    df.columns = [f"V_{i}" for i in range(len(df.columns))]

    display(df)
    return save_hf_dataset(data_dir, dataset_name, [df], start_date=df.index.min(), frequency=frequency)

def save_hf_dataset(data_dir, dataset_name, data, start_date, frequency, num_proc: int = os.cpu_count()):
    """
    Create a Hugging Face Dataset from the generated data.

    """
    data_dict_list = [
        {
            "index": s.index.tolist(),
            "target": [s[col].tolist() for col in s.columns],
        }
        for s in data
    ]

    
    for i, d in enumerate(data_dict_list):
        d["length"] = len(d["index"])
        d["freq"] = frequency
        if len(d["target"]) == 1:
            d["target"] = d["target"][0]
        if isinstance(start_date, pd.Timestamp):
            d["start"] = start_date.strftime('%Y-%m-%d')
        else:
            d["start"] = str(start_date)
        d["start"] = start_date
        d["item_id"] = f"{dataset_name}_sample_{i:06d}"

    dataset = Dataset.from_list(data_dict_list)
    d_sample = dataset[0]
    print(f"Dataset sample: {min(d_sample['index'])}, {max(d_sample['index'])}, length: {d_sample['length']}, freq: {d_sample['freq']}, item_id: {d_sample['item_id']}")
    dataset.save_to_disk(f"{data_dir}/{dataset_name}", num_proc=min(num_proc, len(data)))
