#!/usr/bin/env python
"""
===============================================================================
GIFT-EVAL Experiment Results Aggregation Script
===============================================================================

PURPOSE:
--------
This script aggregates time series forecasting experiment results from
validated_experiment_results.csv, normalizing them against the Seasonal Naive
baseline from all_results.csv. Results are grouped by both MODEL and VARIATE
(MULTI vs UNI) to enable direct comparison of multivariate vs univariate
forecasting performance.

METHODOLOGY (Based on GIFT-EVAL Official Approach):
----------------------------------------------------
1. Load experiment results containing model predictions
2. Load Seasonal Naive baseline results
3. Join baseline to experiment results using 'dataset' as the key
4. Normalize metrics by dividing model performance by baseline performance:
   - normalized_MASE = model_MASE / baseline_MASE
   - normalized_CRPS = model_CRPS / baseline_CRPS
5. Compute arithmetic mean (simple average) of normalized metrics across all datasets
   for each (model, variate) combination
6. Extract dataset metadata (frequency, horizon) from dataset string

INTERPRETATION:
---------------
- normalized_MASE < 1.0: Model outperforms Seasonal Naive baseline
- normalized_MASE > 1.0: Model underperforms Seasonal Naive baseline
- normalized_MASE = 1.0: Model performs equivalently to baseline

The arithmetic mean (simple average) is used for aggregation across datasets.

KEY METRICS:
------------
- MASE[0.5]: Mean Absolute Scaled Error at median (50th percentile)
- CRPS: Continuous Ranked Probability Score (mean_weighted_sum_quantile_loss)

INPUT FILES:
------------
1. validated_experiment_results.csv
   - Contains: dataset, model, variate (MULTI/UNI), eval_metrics/*, etc.
   - Key columns:
     * dataset: Format "dataset_name/frequency/horizon"
     * model: Model identifier (e.g., "Chronos2-GiftEval")
     * model_base: Base model name (e.g., "Chronos2")
     * variate: "MULTI" or "UNI"
     * eval_metrics/MASE[0.5]: Model MASE score
     * eval_metrics/mean_weighted_sum_quantile_loss: Model CRPS score

2. all_results.csv (Seasonal Naive baseline)
   - Contains: dataset, model="Seasonal_Naive", eval_metrics/*, etc.
   - Same dataset format as validated_experiment_results.csv
   - Provides baseline metrics for normalization

OUTPUT:
-------
1. Console output: Aggregated metrics by (model_base, variate) with:
   - Overall normalized MASE and CRPS (arithmetic mean across all datasets)
   - Breakdown by forecast horizon (short/medium/long)
   - Top 5 best and worst performing datasets
   - Missing baseline warnings (if any)

2. Optional summary CSV (--summary-output flag): Aggregated metrics table with:
   - Model, Variate, Datasets (count)
   - Norm_MASE, Norm_CRPS (arithmetic mean across all datasets)
   - One row per (model_base, variate) combination

3. Optional detailed CSV (--output flag): Per-dataset results with:
   - dataset, freq, horizon, model_base, variate
   - eval_metrics/MASE[0.5], eval_metrics/mean_weighted_sum_quantile_loss
   - normalized_MASE, normalized_CRPS

DATA VALIDATION:
----------------
The script validates:
1. All datasets in experiment results have corresponding baseline entries
2. Missing baselines are flagged with WARNING messages
3. Only matched dataset entries are included in aggregation
4. Geometric mean calculations handle edge cases gracefully

USAGE:
------
Basic usage (console output only):
    python scripts/aggregate_validated_experiment_results.py

Save summary aggregated metrics to CSV:
    python scripts/aggregate_validated_experiment_results.py --summary-output summary.csv

Save detailed per-dataset results to CSV:
    python scripts/aggregate_validated_experiment_results.py --output detailed_results.csv

Save both summary and detailed results:
    python scripts/aggregate_validated_experiment_results.py \
        --summary-output summary.csv \
        --output detailed_results.csv

Suppress detailed console output:
    python scripts/aggregate_validated_experiment_results.py --quiet

Specify custom input paths:
    python scripts/aggregate_validated_experiment_results.py \
        --experiment-results path/to/validated_experiment_results.csv \
        --baseline-results path/to/all_results.csv

Example with all options:
    python scripts/aggregate_validated_experiment_results.py \
        --experiment-results results_shared/pipeline_result/validated_experiment_results.csv \
        --baseline-results results_shared/seasonal_naive/all_results.csv \
        --output results_shared/pipeline_result/aggregated_detailed_results.csv \
        --summary-output results_shared/pipeline_result/aggregated_summary_metrics.csv

EXAMPLE OUTPUT:
---------------
Console Output:
======================================================================
GIFT-EVAL Aggregated Results
======================================================================
Model: Chronos2 | Variate: MULTI | Datasets: 16
  Normalized MASE (MASE_Rank):  0.8542
  Normalized CRPS (CRPS_Rank):  0.7123

Model: Chronos2 | Variate: UNI | Datasets: 18
  Normalized MASE (MASE_Rank):  0.8631
  Normalized CRPS (CRPS_Rank):  0.7234

======================================================================
Breakdown by Forecast Horizon (Chronos2 - MULTI):
======================================================================
SHORT horizon (8 datasets):
  Normalized MASE: 0.8234
  Normalized CRPS: 0.6891

Summary CSV Output (--summary-output summary.csv):
Model      Variate  Datasets  Norm_MASE  Norm_CRPS
Chronos2   MULTI    16        0.8542     0.7123
Chronos2   UNI      18        0.8631     0.7234
Moirai     MULTI    16        0.8912     0.7456
Moirai     UNI      18        0.9021     0.7567

DEPENDENCIES:
-------------
- pandas >= 1.3.0
- numpy >= 1.21.0

AUTHOR:
-------
Generated for timeseries-research project
Date: 2026-02-03

===============================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import sys


def geo_mean(iterable):
    """
    Calculate geometric mean.

    The geometric mean is computed as: (x1 * x2 * ... * xn)^(1/n)
    This is more robust than arithmetic mean for ratio/multiplicative data.

    Parameters:
    -----------
    iterable : array-like
        Values to compute geometric mean over

    Returns:
    --------
    float : Geometric mean of the values
    """
    a = np.array(iterable)
    if len(a) == 0:
        return np.nan
    if np.any(a <= 0):
        print("WARNING: Non-positive values detected in geometric mean calculation")
        a = a[a > 0]  # Filter out non-positive values
        if len(a) == 0:
            return np.nan
    return a.prod() ** (1.0 / len(a))


def load_and_validate_data(experiment_path, baseline_path, verbose=True):
    """
    Load and validate experiment and baseline data.

    Parameters:
    -----------
    experiment_path : str or Path
        Path to validated_experiment_results.csv
    baseline_path : str or Path
        Path to seasonal naive all_results.csv
    verbose : bool
        Whether to print validation messages

    Returns:
    --------
    tuple : (experiment_df, baseline_df)
    """
    if verbose:
        print(f"Loading experiment results from: {experiment_path}")
    experiment_df = pd.read_csv(experiment_path)

    if verbose:
        print(f"Loading baseline results from: {baseline_path}")
    baseline_df = pd.read_csv(baseline_path)

    # Strip whitespace from column names (baseline CSV has inconsistent spacing)
    experiment_df.columns = experiment_df.columns.str.strip()
    baseline_df.columns = baseline_df.columns.str.strip()

    # Strip whitespace from dataset values (baseline has trailing spaces)
    experiment_df["dataset"] = experiment_df["dataset"].str.strip()
    baseline_df["dataset"] = baseline_df["dataset"].str.strip()

    # Validate required columns
    required_exp_cols = [
        "dataset",
        "model",
        "model_base",
        "variate",
        "eval_metrics/MASE[0.5]",
        "eval_metrics/mean_weighted_sum_quantile_loss",
    ]
    required_base_cols = [
        "dataset",
        "eval_metrics/MASE[0.5]",
        "eval_metrics/mean_weighted_sum_quantile_loss",
    ]

    missing_exp = [col for col in required_exp_cols if col not in experiment_df.columns]
    missing_base = [col for col in required_base_cols if col not in baseline_df.columns]

    if missing_exp:
        raise ValueError(
            f"Missing required columns in experiment results: {missing_exp}"
        )
    if missing_base:
        raise ValueError(
            f"Missing required columns in baseline results: {missing_base}"
        )

    if verbose:
        print(f"✓ Loaded {len(experiment_df)} experiment results")
        print(f"✓ Loaded {len(baseline_df)} baseline results")
        print(f"✓ Experiment models: {sorted(experiment_df['model_base'].unique())}")
        print(f"✓ Experiment variates: {sorted(experiment_df['variate'].unique())}")

    return experiment_df, baseline_df


def merge_with_baseline(experiment_df, baseline_df, verbose=True):
    """
    Merge experiment results with seasonal naive baseline.

    This function performs a left join to match each experiment result
    with its corresponding baseline, flagging any missing baselines.

    NOTE: The baseline dataset names use uppercase frequency codes (H, D, W)
    while experiment results use lowercase (h, d, w). We normalize both to
    lowercase for matching.

    Parameters:
    -----------
    experiment_df : pd.DataFrame
        Experiment results
    baseline_df : pd.DataFrame
        Baseline results
    verbose : bool
        Whether to print merge diagnostics

    Returns:
    --------
    pd.DataFrame : Merged dataframe with baseline columns suffixed '_baseline'
    """
    # Prepare baseline columns with suffix
    baseline_cols = [
        "dataset",
        "eval_metrics/MASE[0.5]",
        "eval_metrics/mean_weighted_sum_quantile_loss",
    ]
    baseline_subset = baseline_df[baseline_cols].copy()

    # Normalize dataset names to lowercase for matching
    # (baseline uses uppercase frequency codes, experiments use lowercase)
    experiment_df = experiment_df.copy()
    baseline_subset = baseline_subset.copy()

    experiment_df["dataset_normalized"] = experiment_df["dataset"].str.lower()
    baseline_subset["dataset_normalized"] = baseline_subset["dataset"].str.lower()

    # Merge on normalized dataset name
    merged_df = experiment_df.merge(
        baseline_subset[
            [
                "dataset_normalized",
                "eval_metrics/MASE[0.5]",
                "eval_metrics/mean_weighted_sum_quantile_loss",
            ]
        ],
        on="dataset_normalized",
        how="left",
        suffixes=("", "_baseline"),
    )

    # Drop the temporary normalized column
    merged_df = merged_df.drop(columns=["dataset_normalized"])

    # Check for missing baselines
    missing_baseline = merged_df[merged_df["eval_metrics/MASE[0.5]_baseline"].isna()]

    if len(missing_baseline) > 0:
        print(f"\n{'='*70}")
        print(
            f"WARNING: {len(missing_baseline)} experiment results have no baseline match!"
        )
        print(f"{'='*70}")
        print("Missing baselines for datasets:")
        for dataset in missing_baseline["dataset"].unique():
            print(f"  - {dataset}")
        print(
            f"\nThese {len(missing_baseline)} results will be excluded from aggregation."
        )
        print(f"{'='*70}\n")

        # Remove rows with missing baselines
        merged_df = merged_df[
            merged_df["eval_metrics/MASE[0.5]_baseline"].notna()
        ].copy()

    if verbose and len(missing_baseline) == 0:
        print(f"✓ All {len(merged_df)} experiment results matched with baseline")
    elif verbose:
        print(
            f"✓ {len(merged_df)} of {len(experiment_df)} experiment results matched with baseline"
        )

    return merged_df


def filter_datasets_with_both_modes(merged_df, verbose=True):
    """
    Filter to keep only datasets that exist in both MULTI and UNI modes for each model.

    This ensures apples-to-apples comparison between multivariate and univariate forecasting.

    Parameters:
    -----------
    merged_df : pd.DataFrame
        Merged dataframe with model and baseline metrics
    verbose : bool
        Whether to print filtering diagnostics

    Returns:
    --------
    pd.DataFrame : Filtered dataframe with only datasets present in both modes
    """
    merged_df = merged_df.copy()

    # For each model_base, find datasets that exist in both MULTI and UNI
    filtered_dfs = []

    for model_base in merged_df["model_base"].unique():
        model_df = merged_df[merged_df["model_base"] == model_base]

        # Get datasets for MULTI and UNI
        multi_datasets = set(
            model_df[model_df["variate"] == "MULTI"]["dataset"].unique()
        )
        uni_datasets = set(model_df[model_df["variate"] == "UNI"]["dataset"].unique())

        # Find intersection (datasets in both modes)
        common_datasets = multi_datasets & uni_datasets

        if verbose:
            print(f"\nModel: {model_base}")
            print(f"  MULTI datasets: {len(multi_datasets)}")
            print(f"  UNI datasets: {len(uni_datasets)}")
            print(f"  Common datasets (kept): {len(common_datasets)}")

            if len(multi_datasets - common_datasets) > 0:
                print(
                    f"  Removed MULTI-only datasets: {sorted(multi_datasets - common_datasets)}"
                )
            if len(uni_datasets - common_datasets) > 0:
                print(
                    f"  Removed UNI-only datasets: {sorted(uni_datasets - common_datasets)}"
                )

        # Keep only common datasets for this model
        model_filtered = model_df[model_df["dataset"].isin(common_datasets)]
        filtered_dfs.append(model_filtered)

    result_df = pd.concat(filtered_dfs, ignore_index=True)

    if verbose:
        print(f"\n{'='*70}")
        print(f"Filtered from {len(merged_df)} to {len(result_df)} records")
        print("(Kept only datasets existing in both MULTI and UNI modes)")
        print(f"{'='*70}")

    return result_df


def compute_normalized_metrics(merged_df):
    """
    Compute normalized metrics by dividing model metrics by baseline.

    Parameters:
    -----------
    merged_df : pd.DataFrame
        Merged dataframe with model and baseline metrics

    Returns:
    --------
    pd.DataFrame : Dataframe with added normalized_MASE and normalized_CRPS columns
    """
    merged_df = merged_df.copy()

    # Normalize MASE
    merged_df["normalized_MASE"] = (
        merged_df["eval_metrics/MASE[0.5]"]
        / merged_df["eval_metrics/MASE[0.5]_baseline"]
    )

    # Normalize CRPS
    merged_df["normalized_CRPS"] = (
        merged_df["eval_metrics/mean_weighted_sum_quantile_loss"]
        / merged_df["eval_metrics/mean_weighted_sum_quantile_loss_baseline"]
    )

    # Extract metadata from dataset string (format: "name/freq/horizon")
    merged_df["freq"] = merged_df["dataset"].str.split("/").str[1]
    merged_df["horizon"] = merged_df["dataset"].str.split("/").str[2]

    return merged_df


def compute_multi_win_rates(merged_df, verbose=True):
    """
    Compute multivariate vs univariate win rates and add win indicators.

    For each model and dataset, determines whether MULTI mode outperforms UNI mode.
    A "win" means lower normalized metric (better performance).

    Parameters:
    -----------
    merged_df : pd.DataFrame
        Merged dataframe with normalized metrics for both MULTI and UNI
    verbose : bool
        Whether to print diagnostics

    Returns:
    --------
    tuple : (merged_df_with_indicators, win_rates_dict)
        - merged_df_with_indicators: DataFrame with added 'multi_wins_mase' and 'multi_wins_crps' columns
        - win_rates_dict: Dictionary mapping model_base to win rate metrics
    """
    merged_df = merged_df.copy()

    # Create pivot tables to compare MULTI vs UNI for each dataset
    win_rates = {}

    # Initialize win indicator columns
    merged_df["multi_wins_mase"] = False
    merged_df["multi_wins_crps"] = False

    for model_base in merged_df["model_base"].unique():
        model_df = merged_df[merged_df["model_base"] == model_base]

        # Pivot to compare MULTI vs UNI for each dataset
        mase_pivot = model_df.pivot_table(
            index="dataset",
            columns="variate",
            values="normalized_MASE",
            aggfunc="first",
        )

        crps_pivot = model_df.pivot_table(
            index="dataset",
            columns="variate",
            values="normalized_CRPS",
            aggfunc="first",
        )

        # Compute win indicators (MULTI < UNI means MULTI wins)
        mase_multi_wins = mase_pivot["MULTI"] < mase_pivot["UNI"]
        crps_multi_wins = crps_pivot["MULTI"] < crps_pivot["UNI"]

        # Compute win rates
        mase_win_rate = mase_multi_wins.mean()
        crps_win_rate = crps_multi_wins.mean()

        win_rates[model_base] = {
            "MASE_Multi_Win_Rate": mase_win_rate,
            "CRPS_Multi_Win_Rate": crps_win_rate,
            "num_datasets": len(mase_pivot),
        }

        # Add win indicators to the main dataframe
        for dataset in mase_multi_wins.index:
            mask_multi = (
                (merged_df["model_base"] == model_base)
                & (merged_df["dataset"] == dataset)
                & (merged_df["variate"] == "MULTI")
            )
            mask_uni = (
                (merged_df["model_base"] == model_base)
                & (merged_df["dataset"] == dataset)
                & (merged_df["variate"] == "UNI")
            )

            # Set win indicators for both MULTI and UNI rows
            merged_df.loc[mask_multi, "multi_wins_mase"] = mase_multi_wins[dataset]
            merged_df.loc[mask_uni, "multi_wins_mase"] = mase_multi_wins[dataset]
            merged_df.loc[mask_multi, "multi_wins_crps"] = crps_multi_wins[dataset]
            merged_df.loc[mask_uni, "multi_wins_crps"] = crps_multi_wins[dataset]

        if verbose:
            print(f"\n{model_base} Win Rates:")
            print(
                f"  MASE: {mase_win_rate:.4f} ({int(mase_win_rate * len(mase_pivot))}/{len(mase_pivot)} datasets)"
            )
            print(
                f"  CRPS: {crps_win_rate:.4f} ({int(crps_win_rate * len(crps_pivot))}/{len(crps_pivot)} datasets)"
            )

    return merged_df, win_rates


def aggregate_by_model_variate(merged_df, model_base, variate, verbose=True):
    """
    Aggregate metrics for a specific (model_base, variate) combination.

    Parameters:
    -----------
    merged_df : pd.DataFrame
        Merged dataframe with normalized metrics
    model_base : str
        Base model name (e.g., "Chronos2")
    variate : str
        Variate type ("MULTI" or "UNI")
    verbose : bool
        Whether to print detailed results

    Returns:
    --------
    dict : Dictionary with aggregated metrics
    """
    # Filter for this model-variate combination
    subset = merged_df[
        (merged_df["model_base"] == model_base) & (merged_df["variate"] == variate)
    ].copy()

    if len(subset) == 0:
        if verbose:
            print(f"WARNING: No data found for {model_base} - {variate}")
        return None

    # Compute overall arithmetic means
    overall_mase = np.mean(subset["normalized_MASE"].to_numpy())
    overall_crps = np.mean(subset["normalized_CRPS"].to_numpy())

    results = {
        "model_base": model_base,
        "variate": variate,
        "num_datasets": len(subset),
        "overall_normalized_MASE": overall_mase,
        "overall_normalized_CRPS": overall_crps,
        "horizon_breakdown": {},
        "subset_df": subset,
    }

    # Compute breakdown by horizon
    for horizon in ["short", "medium", "long"]:
        horizon_df = subset[subset["horizon"] == horizon]
        if len(horizon_df) > 0:
            results["horizon_breakdown"][horizon] = {
                "num_datasets": len(horizon_df),
                "normalized_MASE": np.mean(horizon_df["normalized_MASE"].to_numpy()),
                "normalized_CRPS": np.mean(horizon_df["normalized_CRPS"].to_numpy()),
            }

    # Print results
    if verbose:
        print(f"\n{'='*70}")
        print("GIFT-EVAL Aggregated Results")
        print(f"{'='*70}")
        print(
            f"Model: {model_base} | Variate: {variate} | Datasets: {results['num_datasets']}"
        )
        print(f"  Normalized MASE (MASE_Rank):  {overall_mase:.4f}")
        print(f"  Normalized CRPS (CRPS_Rank):  {overall_crps:.4f}")
        print("\nInterpretation:")
        print("  - Values < 1.0 indicate the model beats Seasonal Naive baseline")
        print(
            "  - Values > 1.0 indicate the model is worse than Seasonal Naive baseline"
        )

        # Breakdown by horizon
        if results["horizon_breakdown"]:
            print(f"\n{'='*70}")
            print(f"Breakdown by Forecast Horizon ({model_base} - {variate}):")
            print(f"{'='*70}")
            for horizon in ["short", "medium", "long"]:
                if horizon in results["horizon_breakdown"]:
                    hdata = results["horizon_breakdown"][horizon]
                    print(
                        f"\n{horizon.upper()} horizon ({hdata['num_datasets']} datasets):"
                    )
                    print(f"  Normalized MASE: {hdata['normalized_MASE']:.4f}")
                    print(f"  Normalized CRPS: {hdata['normalized_CRPS']:.4f}")

        # Top 5 best and worst datasets by normalized MASE
        print(f"\n{'='*70}")
        print(
            f"Top 5 Best Performing Datasets ({model_base} - {variate}, by normalized MASE):"
        )
        print(f"{'='*70}")
        best_datasets = subset.nsmallest(5, "normalized_MASE")[
            ["dataset", "freq", "horizon", "eval_metrics/MASE[0.5]", "normalized_MASE"]
        ]
        print(best_datasets.to_string(index=False))

        print(f"\n{'='*70}")
        print(
            f"Top 5 Worst Performing Datasets ({model_base} - {variate}, by normalized MASE):"
        )
        print(f"{'='*70}")
        worst_datasets = subset.nlargest(5, "normalized_MASE")[
            ["dataset", "freq", "horizon", "eval_metrics/MASE[0.5]", "normalized_MASE"]
        ]
        print(worst_datasets.to_string(index=False))
        print(f"{'='*70}\n")

    return results


def aggregate_all_results(
    experiment_path,
    baseline_path,
    output_path=None,
    summary_output_path=None,
    verbose=True,
):
    """
    Main aggregation function that processes all model-variate combinations.

    Parameters:
    -----------
    experiment_path : str or Path
        Path to validated_experiment_results.csv
    baseline_path : str or Path
        Path to seasonal naive all_results.csv
    output_path : str or Path, optional
        Path to save detailed results CSV
    summary_output_path : str or Path, optional
        Path to save summary aggregated metrics CSV
    verbose : bool
        Whether to print detailed results

    Returns:
    --------
    tuple : (summary_results_list, merged_df)
        - summary_results_list: List of aggregated results dicts
        - merged_df: Full merged dataframe with all computations
    """
    # Load and validate data
    experiment_df, baseline_df = load_and_validate_data(
        experiment_path, baseline_path, verbose
    )

    # Merge with baseline
    merged_df = merge_with_baseline(experiment_df, baseline_df, verbose)

    # Filter to keep only datasets that exist in both MULTI and UNI modes
    merged_df = filter_datasets_with_both_modes(merged_df, verbose)

    # Compute normalized metrics
    merged_df = compute_normalized_metrics(merged_df)

    # Compute multivariate win rates
    merged_df, win_rates_dict = compute_multi_win_rates(merged_df, verbose)

    # Get unique model-variate combinations
    combinations = (
        merged_df.groupby(["model_base", "variate"])
        .size()
        .reset_index()[["model_base", "variate"]]
    )

    if verbose:
        print(f"\n{'='*70}")
        print(f"Processing {len(combinations)} model-variate combinations:")
        print(f"{'='*70}")
        for _, row in combinations.iterrows():
            print(f"  - {row['model_base']} / {row['variate']}")
        print(f"{'='*70}")

    # Aggregate each combination
    summary_results = []
    for _, row in combinations.iterrows():
        result = aggregate_by_model_variate(
            merged_df, row["model_base"], row["variate"], verbose
        )
        if result:
            summary_results.append(result)

    # Create summary table
    if summary_results:
        summary_rows = []
        for r in summary_results:
            row = {
                "Model": r["model_base"],
                "Variate Mode": r["variate"],
                "MASE": r["overall_normalized_MASE"],
                "CRPS": r["overall_normalized_CRPS"],
            }
            # Add win rates for the first occurrence of each model (avoid duplication)
            if r["variate"] == "MULTI" and r["model_base"] in win_rates_dict:
                row["MASE_Multi_Win_Rate"] = win_rates_dict[r["model_base"]][
                    "MASE_Multi_Win_Rate"
                ]
                row["CRPS_Multi_Win_Rate"] = win_rates_dict[r["model_base"]][
                    "CRPS_Multi_Win_Rate"
                ]
            elif r["variate"] == "UNI" and r["model_base"] in win_rates_dict:
                # For UNI rows, also include win rates for consistency
                row["MASE_Multi_Win_Rate"] = win_rates_dict[r["model_base"]][
                    "MASE_Multi_Win_Rate"
                ]
                row["CRPS_Multi_Win_Rate"] = win_rates_dict[r["model_base"]][
                    "CRPS_Multi_Win_Rate"
                ]
            summary_rows.append(row)

        summary_df = pd.DataFrame(summary_rows)

        # Reorder columns: Model, Variate Mode, MASE, MASE_Multi_Win_Rate, CRPS, CRPS_Multi_Win_Rate
        column_order = [
            "Model",
            "Variate Mode",
            "MASE",
            "MASE_Multi_Win_Rate",
            "CRPS",
            "CRPS_Multi_Win_Rate",
        ]
        summary_df = summary_df[column_order]

        if verbose:
            print(f"\n{'='*70}")
            print("SUMMARY TABLE: All Model-Variate Combinations")
            print(f"{'='*70}")
            # Create a copy with formatted values for display
            display_df = summary_df.copy()
            display_df["MASE"] = display_df["MASE"].apply(lambda x: f"{x:.4f}")
            display_df["CRPS"] = display_df["CRPS"].apply(lambda x: f"{x:.4f}")
            # Format win rates if present
            if "MASE_Multi_Win_Rate" in display_df.columns:
                display_df["MASE_Multi_Win_Rate"] = display_df[
                    "MASE_Multi_Win_Rate"
                ].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "")
            if "CRPS_Multi_Win_Rate" in display_df.columns:
                display_df["CRPS_Multi_Win_Rate"] = display_df[
                    "CRPS_Multi_Win_Rate"
                ].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "")
            print(display_df.to_string(index=False))
            print(f"{'='*70}\n")

        # Save summary table if requested
        if summary_output_path:
            # Round numeric columns to 4 decimal places using float_format
            summary_df.to_csv(summary_output_path, index=False, float_format="%.4f")
            print(f"✓ Summary aggregated metrics saved to: {summary_output_path}\n")

    # Save detailed results if requested
    if output_path:
        output_df = merged_df[
            [
                "dataset",
                "freq",
                "horizon",
                "model_base",
                "variate",
                "eval_metrics/MASE[0.5]",
                "eval_metrics/mean_weighted_sum_quantile_loss",
                "normalized_MASE",
                "normalized_CRPS",
                "multi_wins_mase",
                "multi_wins_crps",
            ]
        ].copy()
        # Save with 4 decimal places for all float columns
        output_df.to_csv(output_path, index=False, float_format="%.4f")
        print(f"✓ Detailed results saved to: {output_path}\n")

    return summary_results, merged_df


def main():
    """Command-line interface for the aggregation script."""
    parser = argparse.ArgumentParser(
        description="Aggregate GIFT-EVAL experiment results by model and variate",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default paths
  python scripts/aggregate_validated_experiment_results.py

  # Save detailed results to CSV
  python scripts/aggregate_validated_experiment_results.py --output aggregated_results.csv

  # Save summary aggregated metrics to CSV
  python scripts/aggregate_validated_experiment_results.py --summary-output summary_metrics.csv

  # Save both detailed and summary results
  python scripts/aggregate_validated_experiment_results.py --output detailed.csv --summary-output summary.csv

  # Suppress detailed output
  python scripts/aggregate_validated_experiment_results.py --quiet

  # Custom input paths
  python scripts/aggregate_validated_experiment_results.py \\
      --experiment-results path/to/validated_experiment_results.csv \\
      --baseline-results path/to/all_results.csv

  # A complete example with all options:
  python scripts/aggregate_validated_experiment_results.py \
        --experiment-results results_shared/pipeline_result/validated_experiment_results.csv \
        --baseline-results results_shared/seasonal_naive/all_results.csv \
        --output results_shared/pipeline_result/aggregated_detailed_results.csv \
        --summary-output results_shared/pipeline_result/aggregated_summary_metrics.csv
        """,
    )

    parser.add_argument(
        "--experiment-results",
        type=str,
        default=None,
        help="Path to validated_experiment_results.csv (default: auto-detect in results_shared/)",
    )

    parser.add_argument(
        "--baseline-results",
        type=str,
        default=None,
        help="Path to seasonal naive all_results.csv (default: auto-detect in results_shared/)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save detailed aggregation results CSV (optional)",
    )

    parser.add_argument(
        "--summary-output",
        type=str,
        default=None,
        help="Path to save summary aggregated metrics CSV (optional)",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress detailed output, only show summary table",
    )

    args = parser.parse_args()

    # Auto-detect paths if not provided
    if args.experiment_results is None:
        script_dir = Path(__file__).parent
        args.experiment_results = (
            script_dir.parent
            / "results_shared"
            / "pipeline_result"
            / "validated_experiment_results.csv"
        )

    if args.baseline_results is None:
        script_dir = Path(__file__).parent
        args.baseline_results = (
            script_dir.parent / "results_shared" / "seasonal_naive" / "all_results.csv"
        )

    # Validate input files exist
    if not Path(args.experiment_results).exists():
        print(f"ERROR: Experiment results file not found: {args.experiment_results}")
        sys.exit(1)

    if not Path(args.baseline_results).exists():
        print(f"ERROR: Baseline results file not found: {args.baseline_results}")
        sys.exit(1)

    # Run aggregation
    try:
        summary_results, merged_df = aggregate_all_results(
            args.experiment_results,
            args.baseline_results,
            output_path=args.output,
            summary_output_path=args.summary_output,
            verbose=not args.quiet,
        )

        print("✓ Aggregation completed successfully!")

    except Exception as e:
        print("ERROR: Aggregation failed with error:")
        print(f"  {type(e).__name__}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
