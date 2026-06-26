"""
Utility helpers for executing GIFTEval experiments on a per-dataset basis.

This module provides a way to iterate through every dataset declared in an
experiment YAML, run the existing ``TsPipeline`` once per dataset, persist the
successful results with descriptive filenames, and record any failures in a log
and summary file.  A final merged CSV containing all successful results is also
generated for easy downstream analysis.
"""

from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from .runner import TsPipeline
from .utils import load_config

LOGGER_NAME = "ts_toolkit.pipeline.dataset_batch_runner"


def _slugify_dataset_name(dataset_cfg: Dict[str, Any]) -> str:
    """
    Create a filesystem-friendly slug for a dataset configuration entry.

    Example:
        {"name": "bitbrains_fast_storage/5T", "term": "long"}
            -> "bitbrains_fast_storage_5T__long"
    """
    name = dataset_cfg.get("name", "unknown").strip()
    term = dataset_cfg.get("term", "unknown").strip()
    slug = name.replace("/", "_").replace(" ", "_").replace(":", "-").replace("\\", "_")
    return f"{slug}__{term}"


def _prepare_logger(log_file: Path) -> logging.Logger:
    """
    Configure and return a logger that writes both to stdout and a logfile.
    """
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)

    if logger.handlers:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()

    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def run_datasets_from_config(config_path: str) -> dict[str, Any]:
    """
    Execute ``TsPipeline`` one dataset at a time using the configuration file.

    Args:
        config_path: Relative path (from ``REPO_ROOT``) to the experiment config.

    Returns:
        A dictionary summarizing the execution with keys:
            - ``successes``: list of dataset slugs that completed.
            - ``failures``: list of dicts with dataset metadata and errors.
            - ``dataset_runs``: detailed per-dataset run metadata including runtime.
            - ``combined_results_path``: path to the merged CSV if created.
            - ``log_path``: path to the execution log file.
    """
    base_config = load_config(config_path=config_path)
    datasets: Iterable[Dict[str, Any]] = deepcopy(base_config.datasets)

    experiment_root = Path(base_config.paths.output_path)
    experiment_root.mkdir(parents=True, exist_ok=True)

    combined_output_dir = experiment_root / base_config.forecaster.alias
    combined_output_dir.mkdir(parents=True, exist_ok=True)

    log_path = experiment_root / "dataset_run.log"
    logger = _prepare_logger(log_path)

    logger.info("Starting per-dataset execution for %s", config_path)
    logger.info("Results will be stored under %s", experiment_root)

    successes: List[str] = []
    failure_records: List[Dict[str, Any]] = []
    result_frames: List[pd.DataFrame] = []
    dataset_run_records: List[Dict[str, Any]] = []

    for dataset_cfg in datasets:
        slug = _slugify_dataset_name(dataset_cfg)
        logger.info("----")
        dataset_name = dataset_cfg.get("name", "")
        dataset_term = dataset_cfg.get("term", "")
        logger.info("Processing dataset '%s' (slug: %s)", dataset_name, slug)

        dataset_output_dir = experiment_root / slug
        dataset_output_dir.mkdir(parents=True, exist_ok=True)

        dataset_config = deepcopy(base_config)
        dataset_config.datasets = [dataset_cfg]
        dataset_config.experiment_name = f"{base_config.experiment_name}__{slug}"
        dataset_config.paths.output_path = str(dataset_output_dir)

        start_time = time.perf_counter()
        try:
            pipeline = TsPipeline(dataset_config)
            pipeline.run(skip_existing=False)
            dataset_results = pipeline.get_results().copy()
        except Exception as exc:  # noqa: BLE001 - need to capture any failure
            elapsed = round(time.perf_counter() - start_time, 3)
            error_message = repr(exc)
            logger.error(
                "ERROR dataset '%s': %s (%.3fs)", dataset_name, error_message, elapsed
            )
            failure_record = {
                "dataset": dataset_name,
                "term": dataset_term,
                "error": error_message,
                "runtime_seconds": elapsed,
            }
            failure_records.append(failure_record)
            dataset_run_records.append(
                {
                    "dataset": dataset_name,
                    "term": dataset_term,
                    "slug": slug,
                    "status": "failure",
                    "runtime_seconds": elapsed,
                    "error": error_message,
                }
            )
            continue

        if dataset_results.empty:
            elapsed = round(time.perf_counter() - start_time, 3)
            error_message = "No results returned"
            logger.warning(
                "WARN dataset '%s' produced no rows; marking as failure. (%.3fs)",
                dataset_name,
                elapsed,
            )
            failure_record = {
                "dataset": dataset_name,
                "term": dataset_term,
                "error": error_message,
                "runtime_seconds": elapsed,
            }
            failure_records.append(failure_record)
            dataset_run_records.append(
                {
                    "dataset": dataset_name,
                    "term": dataset_term,
                    "slug": slug,
                    "status": "failure",
                    "runtime_seconds": elapsed,
                    "error": error_message,
                }
            )
            continue

        results_dir = dataset_output_dir / dataset_config.forecaster.alias
        results_dir.mkdir(parents=True, exist_ok=True)

        dataset_results = dataset_results.assign(dataset_slug=slug)
        results_path = results_dir / f"{slug}_results.csv"
        dataset_results.to_csv(results_path, index=False)

        successes.append(slug)
        result_frames.append(dataset_results)
        elapsed = round(time.perf_counter() - start_time, 3)
        logger.info(
            "OK saved results for '%s' to %s (%.3fs)",
            dataset_name,
            results_path,
            elapsed,
        )
        dataset_run_records.append(
            {
                "dataset": dataset_name,
                "term": dataset_term,
                "slug": slug,
                "status": "success",
                "runtime_seconds": elapsed,
                "results_path": str(results_path),
            }
        )

    combined_results_path: Path | None = None
    if result_frames:
        combined_results = pd.concat(result_frames, ignore_index=True)
        combined_results_path = combined_output_dir / "combined_results.csv"
        combined_results.to_csv(combined_results_path, index=False)
        logger.info("COMBINED results written to %s", combined_results_path)
    else:
        logger.warning("No successful datasets; combined results were not generated.")

    if failure_records:
        failures_path = experiment_root / "failed_datasets.csv"
        failures_df = pd.DataFrame(failure_records)
        failures_df.to_csv(failures_path, index=False)
        logger.info(
            "Logged %d failing dataset(s) to %s",
            len(failure_records),
            failures_path,
        )

    logger.info("----")
    logger.info(
        "Run complete. %d succeeded, %d failed.",
        len(successes),
        len(failure_records),
    )

    return {
        "successes": successes,
        "failures": failure_records,
        "dataset_runs": dataset_run_records,
        "combined_results_path": (
            str(combined_results_path) if combined_results_path else None
        ),
        "log_path": str(log_path),
    }


__all__ = ["run_datasets_from_config"]
