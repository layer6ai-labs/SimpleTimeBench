#!/usr/bin/env python3
"""
Command line runner for executing GIFTEval experiments dataset by dataset.

This script wraps ``run_datasets_from_config`` so users can kick off a long
experiment from the terminal while ensuring that intermediate results, logs,
and a combined summary are persisted.  Paths are resolved relative to the
repository root, allowing execution from any working directory inside the repo.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict


def _ensure_environment(repo_root: Path) -> None:
    """
    Guarantee that required environment variables and import paths are present.
    """
    os.environ.setdefault("REPO_ROOT", str(repo_root))
    default_gift_eval = repo_root / "data" / "gift_eval"
    os.environ.setdefault("GIFT_EVAL", str(default_gift_eval))

    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def _resolve_config_argument(config_arg: str, repo_root: Path) -> str:
    """
    Convert the provided config argument into a repo-relative path.
    """
    config_path = Path(config_arg)
    if not config_path.is_absolute():
        config_path = (Path.cwd() / config_path).resolve()

    if not config_path.exists():
        raise SystemExit(f"Config file not found: {config_path}")

    try:
        relative_path = config_path.relative_to(repo_root)
    except ValueError as exc:
        raise SystemExit(
            f"Config file must be inside the repository: {config_path}"
        ) from exc

    return str(relative_path)


def _write_summary(summary: Dict[str, Any]) -> Path:
    """
    Persist a JSON summary next to the execution log for quick reference.
    """
    log_path_str = summary.get("log_path")
    if not log_path_str:
        summary_path = Path("dataset_run_summary.json")
    else:
        summary_path = Path(log_path_str).with_name("dataset_run_summary.json")

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary_path


def _configure_gpu_device(gpu_identifier: str) -> None:
    """
    Restrict visible CUDA devices to the requested GPU index.
    """
    gpu_identifier = gpu_identifier.strip()
    if not gpu_identifier:
        raise SystemExit("GPU identifier provided to --gpu cannot be empty.")
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_identifier


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Execute an experiment config dataset-by-dataset."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the experiment config (relative to repo root or absolute). E.g. configs/experiments/gifteval.yaml",
    )
    parser.add_argument(
        "--gpu",
        help=(
            "GPU index to expose via CUDA_VISIBLE_DEVICES (e.g. `--gpu 0`). "
            "If omitted, the default CUDA device visibility is used."
        ),
    )
    args = parser.parse_args()

    _ensure_environment(repo_root)

    if args.gpu is not None:
        _configure_gpu_device(args.gpu)
        print(f"Using GPU: {args.gpu}")

    from ts_toolkit.pipeline import run_datasets_from_config

    config_rel = _resolve_config_argument(args.config, repo_root)
    summary = run_datasets_from_config(config_rel)
    summary_path = _write_summary(summary)

    print("Experiment completed.")
    print(f"  Config: {config_rel}")
    print(f"  Successes: {len(summary['successes'])}")
    print(f"  Failures: {len(summary['failures'])}")
    if summary.get("combined_results_path"):
        print(f"  Combined results: {summary['combined_results_path']}")
    print(f"  Log file: {summary['log_path']}")
    print(f"  Summary: {summary_path}")

    if summary["failures"]:
        print("\nFailures:")
        for record in summary["failures"]:
            dataset = record.get("dataset", "<unknown>")
            term = record.get("term", "<unknown>")
            error = record.get("error", "<no error captured>")
            print(f"  - {dataset} ({term}): {error}")


if __name__ == "__main__":
    main()
