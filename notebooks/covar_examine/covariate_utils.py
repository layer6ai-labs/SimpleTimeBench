import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.inspection import permutation_importance
import shap
import seaborn as sns
from statsmodels.tsa.stattools import ccf, grangercausalitytests, adfuller
from sklearn.feature_selection import mutual_info_regression
from itertools import permutations
import os
import warnings
# Suppress verbose output from statsmodels
warnings.filterwarnings("ignore")


def compute_cross_correlation(arr, max_lag=30):
    """
    Compute cross-correlation between variates for various lags.
    
    Parameters:
        arr (numpy.ndarray): Input array of shape (n_variates, n_timestamps).
        max_lag (int): Maximum lag to compute cross-correlation.
        
    Returns:
        numpy.ndarray: Cross-correlation matrix of shape (n_variates, n_variates, 2*max_lag+1).
    """
    n_variates, n_timestamps = arr.shape
    cross_corr_matrix = np.zeros((n_variates, n_variates, 2 * max_lag + 1))
    
    for i in range(n_variates):
        for j in range(n_variates):
            if i != j:
                # Compute cross-correlation for various lags
                lags = np.arange(-max_lag, max_lag + 1)
                for lag in lags:
                    if lag < 0:
                        cross_corr_matrix[i, j, lag + max_lag] = np.corrcoef(arr[i, :lag], arr[j, -lag:])[0, 1]
                    elif lag > 0:
                        cross_corr_matrix[i, j, lag + max_lag] = np.corrcoef(arr[i, lag:], arr[j, :-lag])[0, 1]
                    else:
                        cross_corr_matrix[i, j, lag + max_lag] = np.corrcoef(arr[i, :], arr[j, :])[0, 1]
            else:
                cross_corr_matrix[i, j, :] = 1  # Autocorrelation is always 1
    return cross_corr_matrix

def visualize_cross_correlation(cross_corr_matrix, max_lag):
    """
    Visualize cross-correlation matrix for all variate pairs on a single heatmap.
    
    Parameters:
        cross_corr_matrix (numpy.ndarray): Cross-correlation matrix of shape (n_variates, n_variates, 2*max_lag+1).
        max_lag (int): Maximum lag used in computation.
    """
    n_variates = cross_corr_matrix.shape[0]
    lags = np.arange(-max_lag, max_lag + 1)
    
    # Prepare data for heatmap
    heatmap_data = []
    y_labels = []
    
    for i in range(n_variates):
        for j in range(n_variates):
            if i != j:
                heatmap_data.append(cross_corr_matrix[i, j, :])
                y_labels.append(f"Var {i+1} vs {j+1}")
    
    heatmap_data = np.array(heatmap_data)
    
    # Plot heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        heatmap_data,
        xticklabels=lags,
        yticklabels=y_labels,
        cmap="coolwarm",
        annot=False
    )
    plt.title("Cross-Correlation Heatmap for All Variate Pairs")
    plt.xlabel("Lag")
    plt.ylabel("Variate Pairs")
    plt.show()


def create_lagged_dataset(df: pd.DataFrame, target_name: str, max_lag: int):
    """
    Creates a lagged dataset for supervised learning.

    Args:
        df (pd.DataFrame): The full multivariate time series DataFrame.
        target_name (str): The name of the column to be used as the target (y).
        max_lag (int): The maximum number of lags to create as features.

    Returns:
        tuple: A tuple containing:
            - X (pd.DataFrame): The feature matrix with lagged variates.
            - y (pd.Series): The target series.
    """
    variate_names = df.columns
    features = []
    
    # Create lagged features for all variates
    for name in variate_names:
        for lag in range(1, max_lag + 1):
            features.append(df[name].shift(lag).rename(f"{name}_lag{lag}"))
    
    
    X = pd.concat(features, axis=1)
    y = df[target_name]
    
    # Drop rows with NaN values created by the shifting process
    X = X.iloc[max_lag:]
    y = y.iloc[max_lag:]
    
    return X, y


def check_stationarity(data, variate_names, significance_level=0.05):
    """
    Performs the Augmented Dickey-Fuller test for stationarity on each variate.
    """
    print("--- Checking for Stationarity (ADF Test) ---")
    is_stationary = True
    for i, name in enumerate(variate_names):
        adf_test = adfuller(data[i, :])
        p_value = adf_test[1]
        if p_value > significance_level:
            print(f"Warning: '{name}' is likely non-stationary (p-value: {p_value:.4f}).")
            print("         Granger Causality results may be unreliable.")
            is_stationary = False
        # else:
        #     print(f"'{name}' is likely stationary (p-value: {p_value:.4f}).")
    if not is_stationary:
        print("\nConsider differencing the non-stationary series before analysis.")
    print("-" * 45 + "\n")
    return is_stationary


# --- REPLACE THE ENTIRE analyze_variate_contributions FUNCTION WITH THIS ---
def analyze_variate_contributions(
    ts_data: np.ndarray,
    variate_names: list = None,
    max_lag: int = 10,
    verbose: bool = False
):
    """
    Analyzes a multivariate time series to find contributions between variates.
    Calculates cross-correlation, mutual information, Granger causality,
    permutation importance, and SHAP importance.

    Args:
        ts_data (np.ndarray): The time series data of shape (n_variates, n_timestamps).
        variate_names (list, optional): List of names for the variates. Defaults to V0, V1...
        max_lag (int, optional): The maximum number of lags to consider. Defaults to 10.
        verbose (bool, optional): Whether to print a summary of the results. Defaults to True.

    Returns:
        dict: A dictionary containing results for all calculated metrics.
    """
    n_variates, n_timestamps = ts_data.shape

    if variate_names is None:
        variate_names = [f'V{i}' for i in range(n_variates)]

    df = pd.DataFrame(ts_data.T, columns=variate_names)
    
    # Stationarity Check
    if verbose:
      check_stationarity(ts_data, variate_names)

    # --- Initialize results storage ---
    results = {
        'cross_correlation': pd.DataFrame(np.nan, index=variate_names, columns=variate_names),
        'mutual_information': pd.DataFrame(np.nan, index=variate_names, columns=variate_names),
        'granger_causality': pd.DataFrame(np.nan, index=variate_names, columns=variate_names),
        'permutation_importance': pd.DataFrame(np.nan, index=variate_names, columns=variate_names),
        'shap_importance': pd.DataFrame(np.nan, index=variate_names, columns=variate_names),
    }

    # --- 1. Pairwise Statistical and Info-Theoretic Metrics ---
    for cause_name, effect_name in permutations(variate_names, 2):
        # Cross-Correlation
        ccf_vals = ccf(df[effect_name], df[cause_name], adjusted=True)[1:max_lag+1]
        results['cross_correlation'].loc[effect_name, cause_name] = np.max(np.abs(ccf_vals))
        
        # Mutual Information
        mi_scores = [mutual_info_regression(df[cause_name].values[:-lag].reshape(-1, 1), df[effect_name].values[lag:])[0] for lag in range(1, max_lag + 1)]
        results['mutual_information'].loc[effect_name, cause_name] = np.max(mi_scores) if mi_scores else 0

        # Granger Causality
        try:
            gc_test = grangercausalitytests(df[[effect_name, cause_name]], maxlag=max_lag, verbose=verbose)
        except Exception as e:
            print(f"Error in Granger Causality test for {effect_name} -> {cause_name}: {e}")
            gc_test = None
            
        min_p_value = np.min([gc_test[lag][0]['ssr_ftest'][1] for lag in range(1, max_lag + 1)]) if gc_test else np.nan
        results['granger_causality'].loc[effect_name, cause_name] = min_p_value

    # --- 2. Model-Based Importance Metrics (Permutation & SHAP) ---
    # This requires a different loop structure: one model per "effect" variate.
    if verbose:
      print("\n--- Building Models for Permutation and SHAP Importance ---")
      print("(This may take a moment for larger datasets or more variates)...")

    for effect_name in variate_names:
        if verbose:
          print(f"  Building model to predict '{effect_name}'...")
        
        # Create lagged dataset for the current target
        X, y = create_lagged_dataset(df, effect_name, max_lag)
        
        # Train a LightGBM model
        model = lgb.LGBMRegressor(random_state=42, n_estimators=100,  force_col_wise=True, verbose=-1) #
        model.fit(X, y)
        
        # Calculate Permutation Importance
        perm_imp = permutation_importance(model, X, y, n_repeats=5, random_state=42, n_jobs=-1)
        
        # Calculate SHAP Importance
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        # For global importance, take the mean absolute SHAP value per feature
        shap_imp_scores = np.mean(np.abs(shap_values), axis=0)
        
        # Aggregate importances by the original variate name
        for cause_name in variate_names:
            if cause_name == effect_name:
                continue

            # Find all feature indices corresponding to the 'cause_name'
            cause_feature_indices = [i for i, col in enumerate(X.columns) if col.startswith(cause_name)]
            
            # Sum the importances for this cause variate
            total_perm_imp = perm_imp.importances_mean[cause_feature_indices].sum()
            total_shap_imp = shap_imp_scores[cause_feature_indices].sum()
            
            results['permutation_importance'].loc[effect_name, cause_name] = total_perm_imp
            results['shap_importance'].loc[effect_name, cause_name] = total_shap_imp
    
    if verbose:
        print("\n--- Summary of Inter-Variate Contributions ---")
        print("NOTE: For (Cause -> Effect), read tables as (Column -> Row)\n")
        
        print("\n[1] Max Absolute Cross-Correlation (up to {} lags):".format(max_lag))
        print(results['cross_correlation'].round(3))
        
        print("\n[2] Max Mutual Information (up to {} lags):".format(max_lag))
        print(results['mutual_information'].round(3))
        
        print("\n[3] Granger Causality (min p-value up to {} lags):".format(max_lag))
        print(results['granger_causality'].round(3))

        print("\n[4] Aggregated Permutation Importance:")
        print("(Higher value means variate is more important for model predictions)")
        print(results['permutation_importance'].round(3))

        print("\n[5] Aggregated SHAP Importance:")
        print("(Higher value means variate has a larger impact on model output magnitude)")
        print(results['shap_importance'].round(3))

    return results

def plot_analysis_results(results, p_value_threshold=0.05):
    """
    Visualizes the analysis results using five heatmaps.
    """
    # Increase figure size to accommodate 5 plots
    fig, axes = plt.subplots(1, 5, figsize=(35, 6), sharey=True)
    plt.suptitle("Inter-Variate Contribution Analysis", fontsize=20, y=1.02)
    
    common_heatmap_kws = dict(annot=True, fmt=".2f", linewidths=.5)

    # 1. Cross-Correlation
    sns.heatmap(results['cross_correlation'], ax=axes[0], cmap='viridis', **common_heatmap_kws)
    axes[0].set_title("Max Abs Cross-Correlation", fontsize=14)
    axes[0].set_xlabel("Cause Variate", fontsize=12)
    axes[0].set_ylabel("Effect Variate", fontsize=12)

    # 2. Mutual Information
    sns.heatmap(results['mutual_information'], ax=axes[1], cmap='plasma', **common_heatmap_kws)
    axes[1].set_title("Max Mutual Information", fontsize=14)
    axes[1].set_xlabel("Cause Variate", fontsize=12)

    # 3. Granger Causality (Boolean)
    granger_bool = results['granger_causality'] < p_value_threshold
    sns.heatmap(granger_bool, ax=axes[2], cmap='coolwarm', annot=True, cbar=False)
    axes[2].set_title(f"Granger Causality (p < {p_value_threshold})", fontsize=14)
    axes[2].set_xlabel("Cause Variate", fontsize=12)

    # 4. Permutation Importance
    sns.heatmap(results['permutation_importance'], ax=axes[3], cmap='cividis', **common_heatmap_kws)
    axes[3].set_title("Permutation Importance", fontsize=14)
    axes[3].set_xlabel("Cause Variate", fontsize=12)
    
    # 5. SHAP Importance
    sns.heatmap(results['shap_importance'], ax=axes[4], cmap='magma', **common_heatmap_kws)
    axes[4].set_title("SHAP Importance", fontsize=14)
    axes[4].set_xlabel("Cause Variate", fontsize=12)

    # Rotate tick labels for better readability
    for ax in axes:
        ax.tick_params(axis='x', rotation=45)
        ax.tick_params(axis='y', rotation=0)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
