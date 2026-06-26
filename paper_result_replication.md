# Purpose
This document explains how to reproduce the experiment results presented in the paper.

# Multi vs. Uni

## Pipeline Results Generation (Optional)
The pipeline results are already available in `results_shared/pipeline_result`.

If you want to regenerate them, run the corresponding configs in `experiments/multi_vs_uni`. E.g.:

```
python3 scripts/run_gifteval_config.py --config configs/experiments/multi_vs_uni/gifteval_all_terms_chronos2_multi_4096.yaml --gpu 0
```

Results are saved to the `results` folder by default.

Note: If you were to replace the saved results with your re-generated results, the timestamp in the folder needs to be updated accordingly. 

## Result Post-Processing and Aggregation
Run `notebooks/04_merge-results/01_load_and_merge.ipynb` first (top to bottom), then run `notebooks/04_merge-results/02_analysis.ipynb` (top to bottom), and finally execute:

```
python scripts/aggregate_validated_experiment_results.py \
        --experiment-results results_shared/pipeline_result/validated_experiment_results.csv \
        --baseline-results results_shared/seasonal_naive/all_results.csv \
        --output results_shared/pipeline_result/aggregated_detailed_results.csv \
        --summary-output results_shared/pipeline_result/aggregated_summary_metrics.csv
```

The final aggregated results are saved to `results_shared/pipeline_result/aggregated_summary_metrics.csv`.

The appendix plot can be found in `notebooks/04_merge-results/02_analysis.ipynb`.
