import logging
from tqdm.notebook import tqdm
import pandas as pd
import numpy as np
import pandas as pd
import os
import logging
from tqdm.notebook import tqdm
import pandas as pd
import numpy as np
import pandas as pd
import os
import pickle
from datetime import datetime
from ts_toolkit.pipeline.runner import TsPipeline
from ts_toolkit.pipeline.utils import load_config, forecast_output_saver

DISTRIBUTIONS = {
    "Deterministic Trends": ['constant', 'linear', 'exponential', 'staircase'],
    "Harmonic & Periodic": ['sine', 'sawtooth', 'square', 'triangle', 'noise_patch', 'fourier'],
    "Transient Dynamics": ['damped_sine', 'growing_sine', 'chirp'],
    "Autocorrelated Processes": ['ar', 'ma', 'arma', 'sarima', 'garch', 'pink_noise', 'fractional_brownian', 'markov_chain'], 
    "I.I.D. White Noise": ['normal', 'uniform', 'poisson', 'binary', 'student_t', 'lognormal', 'laplace', 'cauchy', 'skew_normal'],
    "Stochastic Drift/Shocks": ['random_walk', 'piecewise_constant', 'gbm', 'intermittent', 'impulse', 'logistic'], 
}

Chronos2_baselines = {
    "Chronos2 (Baseline) - Multivariate": "results/Chronos2-Multivariate-SimpleTime_20260210-134650/Chronos2-Multivariate-SimpleTime",
    # "Chronos2 (Baseline) - Univariate": "results/Chronos2-Univariate-SimpleTime_20260210-135127/Chronos2-Univariate-SimpleTime",
}

# The `METRIC_LIST` dictionary is storing different sets of evaluation metrics for different scenarios
# of the Chronos2 model after finetuning. Each key in the dictionary represents a specific scenario of
# the Chronos2 model after finetuning, and the corresponding value is a list of evaluation metrics
# associated with that scenario.
# scenario. These evaluation metrics are used to assess the performance of the model in each scenario.
METRIC_LIST = {
    "Chronos2 (Finetuned - Bias)": ['MeanBiasRatio'], # ['MeanBiasRatio', 'MeanSkillRatio'],
    "Chronos2 (Finetuned - Norm)": ['MeanAttractionRatio', 'PathVolatilityRatio', 'MAPE'],
    "Chronos2 (Finetuned - CopyPaste)": ['PersistenceSkillRatio', 'MeanSkillRatio', 'MAPE'],
    "Chronos2 (Finetuned - All)": ['PersistenceSkillRatio', 'MeanSkillRatio', 'MAPE'],
}

YELLOW = "#F8E796"
DARK_YELLOW = "#CFA70A"
BLUE = "#227bbb"
SKY_BLUE = '#87ceeb'
NAVY = '#000080'
DARK_RED = '#8b0000'
RED = "#B60707"
FOREST_GREEN = "#064306"
GREEN = "#309930"
GRAY = '#808080'

colors = {
    "input": BLUE, "leading_covariate": SKY_BLUE, "boundary": GRAY,
    "input": BLUE, "leading_covariate": SKY_BLUE, "boundary": GRAY,
    "Chronos2": RED, "Chronos2-Finetuned": GREEN, "Toto": NAVY, "Moirai": DARK_YELLOW
}


def compute_improve(baseline_values, model_values, metric):
    if any(m in metric for m in ['PathVolatilityRatio', 'MeanAttractionRatio']): # the closer to 1 the better
        return (abs(baseline_values - 1) - abs(model_values - 1)) / abs(baseline_values - 1) * 100
    elif any(m in metric for m in ['MeanBiasRatio']): # the closer to 0.5 the better
        return (abs(baseline_values - 0.5) - abs(model_values - 0.5)) / abs(baseline_values - 0.5) * 100
    elif any(m in metric for m in [ # the lower the better
        'PersistenceSkillRatio', 'MeanSkillRatio', 'MSE', 'MAE', 'MAPE', 'sMAPE', 'MASE', 'MSIS', 'quantile_loss', 'ND', 'NormalizedMeanBias', 'RelativeUncertainty'
    ]): # TODO Confirm ND & MSIS
        return (baseline_values - model_values) / baseline_values * 100
    else:
        raise ValueError(f"Unknown metric {metric} for improvement calculation")
    
def generate_final_table(agg_df, model):
    metric_list = METRIC_LIST[model.split(')')[0] + ')']
    model_df = agg_df[[col for col in agg_df.columns if ('eval_metrics' not in col or any(f"/{mt}" in col for mt in metric_list))]]
    model_df = model_df[model_df['model'].isin(["Chronos2 (Baseline)", model])]
    model_df_piv = model_df.pivot(index=[
        'dataset', 'variate', 'num_variates', 'var_mode', 'distribution', 'distribution-family'
    ], columns='model', values=[col for col in model_df.columns if 'eval_metrics' in col])
    for metric, _ in model_df_piv.columns:
        # skip if values are NaN for either baseline or model
        model_df_piv[(metric, '%Improv')] = compute_improve(model_df_piv[(metric, "Chronos2 (Baseline)")].values, model_df_piv[(metric, model)].values, metric)
    # sort columns to have baselin>model>%improv per metric
    df_final = model_df_piv.reindex(columns=[col for metric in sorted(set([col[0] for col in model_df_piv.columns])) for col in [(metric, 'Chronos2 (Baseline)'), (metric, model), (metric, '%Improv')]])
    df_final = df_final.dropna(subset= [col for col in df_final.columns if f'%Improv' in col[1]], how='any')
    return df_final

def groupby(model_df_piv, model, groupby_cols=['variate', 'var_mode', 'distribution'], agg_functions=['mean'], round_digits=2):
    metric_list = METRIC_LIST[model.split(')')[0] + ')']
    df_gb = model_df_piv.groupby(groupby_cols)[[(metric, model) for (metric, model) in model_df_piv.columns if any(f'/{mt}' in metric for mt in metric_list)]]
    if agg_functions == ['mean']:
        df_gb = df_gb.mean()
    elif agg_functions == ['median']:
        df_gb = df_gb.median()
    else:
        df_gb = df_gb.agg(agg_functions)

    return df_gb.round(round_digits)
    
def combine_results(repertoire_dict):
    agg_df = pd.DataFrame()
    for experiment, path in repertoire_dict.items():
        print(f"{experiment}: {path}")
        df = pd.read_csv(os.path.join(path, "all_results.csv"))
        df['model'] = experiment.replace(' - Multivariate', '').replace(' - Univariate', '')# ' - '.join(experiment.split(' - ')[:-1])
        # add a new column called distribution that shows the dataset column final string after / has a match with which of distribution keys in DISTRIBUTIONS dict
        df['distribution-family'] = df['dataset'].apply(lambda x: [k for k, v in DISTRIBUTIONS.items() if x.split('/')[-2] in v][0])
        df['distribution'] = df['dataset'].apply(lambda x: x.split('/')[-2])
        agg_df = pd.concat([agg_df, df], axis=0)
    agg_df = agg_df.reset_index(drop=True)
    agg_df['var_mode'] = agg_df['dataset'].apply(lambda x: x.split('/')[0])
    return agg_df


def save_latex_table(df, index=False):
    latex_table = df.to_latex(
        index=index, 
        caption="", 
        label="",
       column_format='c' * (len(df.columns) + (1 if index else 0)),
        float_format="$.2f\%$"
    )
    latex_table = latex_table.replace('eval_metrics/', '').replace('[512]', '').replace('[0.5]', '').replace('%', r'\%').replace('_', r'-')
    return latex_table
