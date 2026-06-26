"""
Script to generate synthetic time series datasets with various distributions.
"""

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

import pandas as pd
from dotenv import load_dotenv

from datasets import Dataset  # To save in Hugging Face format
from gifteval_dataset import Dataset as GiftEvalDataset
from plot_timeseries import plot_multivariate_timeseries_subplots
from timeseries_generator import TimeSeriesGenerator


@dataclass
class DatasetConfig:
    """Configuration for dataset generation."""

    length: int
    num_samples: int
    num_series: int
    lag_amount: int
    verbose: bool
    nprint: int
    plot_examples: bool
    nplt: int
    storage_env_var: str
    skip_load_check: bool
    num_proc: int = os.cpu_count()
    skip_existing: bool = True


def save_hf_dataset(
    data_dir: str,
    dataset_name: str,
    data: List[pd.DataFrame],
    config: DatasetConfig,
    start_date: str,
    frequency: str,
    num_proc: int = os.cpu_count(),
):
    """Create and save a Hugging Face Dataset from the generated data."""
    data_dict_list = [
        {
            "index": s.index.tolist(),
            "target": [s[col].tolist() for col in s.columns if "V_" in col],
        }
        for s in data
    ]

    for i, d in enumerate(data_dict_list):
        d["length"] = config.length
        d["freq"] = frequency
        if len(d["target"]) == 1:
            d["target"] = d["target"][0]
        d["start"] = start_date
        d["item_id"] = f"{dataset_name}_sample_{i:06d}"

    dataset = Dataset.from_list(data_dict_list)
    dataset.save_to_disk(f"{data_dir}/{dataset_name}", num_proc=min(num_proc, len(data)))


def test_load(ds_name: str, storage_env_var: str):
    """Verify that the saved dataset can be loaded correctly."""
    try:
        dataset = GiftEvalDataset(
            name=ds_name,
            term="short",
            to_univariate=False,
            storage_env_var=storage_env_var,
        )
        print(f"Successfully verified load for {ds_name}")
        print(f"  Frequency: {dataset.freq}")
        print(f"  Windows: {dataset.windows}")
    except Exception as e:
        print(f"FAILED to verify load for {ds_name}: {e}")


def run_generation_pipeline(
    generator: TimeSeriesGenerator,
    config: DatasetConfig,
    univariate: bool,
    distribution: str,
    lagged: bool = False,
):
    """Orchestrates the generation, processing, and saving of a dataset."""
    type_str = "univariate" if univariate else "multivariate"
    if lagged:
        type_str += "_lagged"
    load_dotenv()
    storage_path = os.getenv(config.storage_env_var)
    if not storage_path:
        raise ValueError(f"Environment variable {config.storage_env_var} not set.")
    
    if True:
        # skip if dataset already exists 
        if os.path.exists(os.path.join(storage_path, type_str, distribution)):
            print(f"Dataset {type_str}/{distribution} already exists. Skipping generation.")
            return

    print(f"\nGenerating {type_str} dataset: {distribution}")

    # 1. Generate Data
    if univariate:
        data = generator.create_univariate_dataset(
            num_samples=config.num_samples,
            length=config.length,
            distribution=distribution,
        )
    else:
        data = generator.create_multivariate_dataset(
            num_samples=config.num_samples,
            length=config.length,
            num_series=config.num_series,
            lagged=lagged,
            lag_amount=config.lag_amount,
            distributions=distribution,
        )

    # 2. Verbose Output
    if config.verbose:
        for i, df in enumerate(data[: config.nprint]):
            print(f"Sample {i+1}:\n{df.head()}\n")

    # 3. Plotting
    if config.plot_examples:
        plot_multivariate_timeseries_subplots(
            data, nplt=config.nplt, title=f"{type_str.capitalize()}: {distribution}"
        )

    # 4. Saving
    save_dir = Path(storage_path) / type_str
    save_dir.mkdir(parents=True, exist_ok=True)

    save_hf_dataset(
        data_dir=str(save_dir),
        dataset_name=distribution,
        data=data,
        config=config,
        start_date=generator.start_date,
        frequency=generator.frequency,
        num_proc=config.num_proc or os.cpu_count(),
    )

    # 5. Load Check
    # if not config.skip_load_check:
    #     test_load(f"{type_str}/{distribution}", config.storage_env_var)


def main(args=None):
    """Main entry point to parse arguments and run the generation pipeline."""
    if args is None:
        parser = argparse.ArgumentParser(
            description="Generate synthetic time series datasets."
        )
        parser.add_argument(
            "--length", type=int, default=256, help="Length of each time series"
        )
        parser.add_argument(
            "--num_samples",
            type=int,
            default=100,
            help="Number of samples to generate for each dataset",
        )
        parser.add_argument(
            "--num_series",
            type=int,
            default=2,
            help="Number of series (variables) in multivariate datasets",
        )
        parser.add_argument(
            "--lag_amount",
            type=int,
            default=48,
            help="Number of time steps to lag the covariates",
        )
        parser.add_argument(
            "--skip_load_check",
            action="store_true",
            help="Skip the immediate load check after generation",
        )
        parser.add_argument(
            "--verbose", action="store_true", help="Whether to print dataset samples"
        )
        parser.add_argument(
            "--nprint",
            type=int,
            default=3,
            help="Number of samples to print if verbose",
        )
        parser.add_argument(
            "--plot_examples", action="store_true", help="Whether to plot the datasets"
        )
        parser.add_argument(
            "--nplt",
            type=int,
            default=3,
            help="Number of samples to plot if plot_examples",
        )
        parser.add_argument(
            "--storage_env_var",
            type=str,
            default="SYNTHETIC",
            help="Env var setting path to save datasets",
        )
        parser.add_argument(
            "--num_proc",
            type=int,
            default=os.cpu_count(),
            help="Number of processes to use for saving datasets",
        )
        parser.add_argument(
            "--skip_existing",
            action="store_true",
            help="Skip generation if dataset already exists in storage",
        )
        parser.add_argument(
            "distributions",
            nargs="*",
            default=[
                "constant",
                "linear",
                "exponential",
                "staircase",
                "piecewise_constant",
                "sine",
                "sawtooth",
                "square",
                "triangle",
                "noise_patch",
                "fourier",
                "damped_sine",
                "growing_sine",
                "chirp",
                "ar",
                "ma",
                "arma",
                "sarima",
                "garch",
                "pink_noise",
                "fractional_brownian",
                "markov_chain",
                "normal",
                "uniform",
                "poisson",
                "binary",
                "student_t",
                "lognormal",
                "laplace",
                "cauchy",
                "skew_normal",
                "random_walk",
                "gbm",
                "intermittent",
                "impulse",
                "logistic",
            ],
            help="List of distributions to generate",
        )

        args = parser.parse_args()

    config = DatasetConfig(
        length=args.length,
        num_samples=args.num_samples,
        num_series=args.num_series,
        lag_amount=args.lag_amount,
        verbose=args.verbose,
        nprint=args.nprint,
        plot_examples=args.plot_examples,
        nplt=args.nplt,
        storage_env_var=args.storage_env_var,
        skip_load_check=args.skip_load_check,
        num_proc=args.num_proc,
        skip_existing=args.skip_existing,
    )

    base_seed = 13579
    for distribution in args.distributions:
        
        # Run univariate pipeline
        generator = TimeSeriesGenerator(random_state=base_seed)
        run_generation_pipeline(
            generator, config, univariate=True, distribution=distribution
        )

        # Run multivariate pipeline
        generator = TimeSeriesGenerator(random_state=base_seed)
        run_generation_pipeline(
            generator, config, univariate=False, distribution=distribution, lagged=False
        )

        # Run multivariate lagged pipeline
        generator = TimeSeriesGenerator(random_state=base_seed)
        run_generation_pipeline(
            generator, config, univariate=False, distribution=distribution, lagged=True
        )


if __name__ == "__main__":
    main()
