# SimpleTimeBench

simpletime is a benchmarking suite designed to test time-series models on basic forecasting scenarios. Timeseries can be generated from a number of different generative processes that span a range of difficulties, including deterministic trends, periodic signals, noise from different distributions, and more. 

There are three main types of tasks in simpletime:

1. Univariate Forecasting:
Tests the model's ability to identify fundamental trends, patterns, and noise properties from simple timeseries.
    - Patterns: Trends, periodic oscillations, Autocorrelated processes, noise, etc.
    - Human Intuition: These have a range of difficulties from very easy to impossible.Some are "ruler and pencil" tasks—simply continuing a line or "copy-pasting" a cycle. Others are impossible to predict, such as white noise.

2. Multivariate Forecasting
Tests the model's ability to handle dimensionality without performance degradation.
    - Patterns: Multiple independent univariate series from the same generative distribution predicted simultaneously.
    - Human Intuition: Predicting two independent time series should be no more difficult than predicting one; the model must demonstrate it isn't "distracted" by additional channels.

3. Covariate Support (Future-Leaking)
Tests if a model can identify and exploit external signals that have a direct causal or temporal relationship with the target.
    - Patterns: The target ($Var_0$) is supplemented by covariates that are identical to the target but shifted *backward* in time.
    - Human Intuition: If a covariate tells you exactly what happened tomorrow, the forecast becomes trivial. This simulates an ideal situation of real-world "lead indicators" found in macroeconomics, supply chains, and sensor networks. If models on the multivariate_lagged tasks cannot outperform univariate, it suggests they are not learning to leverage the covariate additional information.

**Generative Regimes**

| Regime | Description | Distributions Included | Assumed Prediction Difficulty |
| :--- | :--- | :--- | :--- |
| **Deterministic Trends** | Low-dimensional, non-repeating growth or states. | `constant`, `linear`, `exponential`, `staircase` | **Trivial**: Solvable with 1-2 parameters via regression. |
| **Harmonic & Periodic** | Fixed-frequency oscillations and synthetic patches. | `sine`, `sawtooth`, `square`, `triangle`, `noise_patch`, `fourier` | **Low**: Predictable via phase/amplitude estimation. |
| **Transient Dynamics** | Signals with time-varying frequency or amplitude. | `damped_sine`, `growing_sine`, `chirp` | **Moderate**: Requires tracking rate-of-change parameters. |
| **Autocorrelated Processes** | Randomness with "memory" or time-dependency. | `ar`, `ma`, `arma`, `sarima`, `garch`, `pink_noise`, `fractional_brownian`, `markov_chain` | **High**: Limited to predicting the conditional mean or variance. |
| **I.I.D. White Noise** | Independent samples from a fixed distribution. | `normal`, `uniform`, `poisson`, `binary`, `student_t`, `lognormal`, `laplace`, `cauchy`, `skew_normal` | **Impossible**: The best prediction is the global mean/median. |
| **Stochastic Drift/Shocks** | Cumulative randomness or sudden impulses. | `random_walk`, `piecewise_constant`, `gbm` (Geometric Brownian), `intermittent`, `impulse`, `logistic` | **Impossible**: Path-dependent; no mean-reversion to exploit. |



# Model Evaluation Scorecard

Below is an example of a model evaluation scorecard for a hypothetical model.

| Task Type | Capability Tested | Result |
| :--- | :--- | :--- |
| Univariate | Can the model extrapolate basic geometry? | Pass: Model identifies linear and exponential growth to acceptable degree |
| Multivariate | Does adding independent variates from the same generating process confuse the model or help it through a form of in context learning? | Fail: Performance dropped when  variates were added. |
| Future-Leaking | Can the model identify perfect temporal leads? | Pass: Model successfully "cheated" using the lagged covariate. |

| Regime | Metric (e.g., MAE) | Benchmark Score | Status |
| :--- | :--- | :--- | :--- |
| **Deterministic Trends** | 0.002 | 99% | ✅ Pass |
| **Harmonic & Periodic** | 0.015 | 95% | ✅ Pass |
| **Transient Dynamics** | 0.120 | 82% | ⚠️ Marginal |
| **Autocorrelated** | 0.450 | 55% | ℹ️ Expected |
| **I.I.D. White Noise** | 1.050 | -- | 🟦 Baseline |
| **Stochastic Drift** | 2.300 | -- | 🟦 Baseline |

Interpretation Guide:

- The "Human Gap": If the model shows high error on Deterministic or Harmonic tasks, the architecture likely has an "inductive bias" problem (e.g., a transformer failing to extrapolate a simple linear slope).

- The "Noise Floor": Error on I.I.D. and Stochastic regimes is expected. These serve as a control group to ensure the model isn't hallucinating patterns in pure randomness.

# Instructions

## Generating and Plotting Data from Notebook

see `create_and_plot_simpletime.ipynb`

## Generating Data from script
1. From the root git directory, set SYNTHETIC env variable in .env to the location you want to save the data to (default `data/simpletime/datasets`)
2. From the root git directory, run `python data/simpletime/simpletime.py`. This will save data to location in #1
3. (Optional:) Set command line params (`python data/simpletime/simpletime.py` to see options). `--plot_examples` will plot sample timeseries from each dataset


    | arg             | default | explanation                                         |
    |-----------------|---------|-----------------------------------------------------|
    | --length        | 256     | Length of each time series                          |
    | --num_samples   | 100     | Number of samples to generate for each dataset      |
    | --num_series    | 2       | Number of series (variables) in multivariate datasets|
    | --lag_amount    | 48      | Number of time steps to lag the covariates          |
    | --verbose       | False   | Whether to print dataset samples                    |
    | --nprint        | 3       | Number of samples to print if verbose               |
    | --plot_examples | False   | Whether to plot the datasets                        |
    | --nplt          | 3       | Number of samples to plot if plot_examples          |