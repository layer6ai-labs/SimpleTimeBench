import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from itertools import islice
import json
import os
from glob import glob
import math

DARK_GREY = "#1c2b34"
BLUE = "#3598ec"
DARK_BLUE = "#0b3c5d"
PURPLE = "#7463e1"
LIGHT_PURPLE = "#d7c3ff"
PINK = "#ff0099"


def plot_forecast(
    dataset,
    forecasts,
    to_univariate: bool = False,
    dataset_label: str | None = None,
    num_plots: int = 5,
    show_plot: bool = True,
):
    figures = []
    plotted = 0
    for data, next_data in zip(dataset, islice(dataset, 1, None)):
        assert data[0]['item_id'] == data[1]['item_id'], f"Data input item_id {data[0]['item_id']} does not match label item_id {data[1]['item_id']}"
        item_id = data[0]['item_id'].split('_dim')[0]
        if to_univariate:
            try:
                if 'dim0' in data[0]['item_id'] or 'dim' not in data[0]['item_id']:
                    input_list, label_list = [data[0]['target']], [data[1]['target']]
                else: # append next variate
                    input_list.append(data[0]['target'])
                    label_list.append(data[1]['target'])
                # Check if the next data exists before accessing it
                if next_data is not None and 'dim' in data[0]['item_id'] and 'dim0' not in next_data[0]['item_id']:
                    continue
            except Exception as e:
                print("data[0]['item_id'] = ", data[0]['item_id'])
                print("data[0] = ", data[0])
                raise e
        else:
            input_list, label_list = data[0]["target"], data[1]["target"] # shape (num_variates, seq_len)

        # only plot num_plots items
        plotted += 1
        if plotted > num_plots:
            break
        # aggregate all forecast variates for the current item_id
        forecast_list = []
        for fcst in forecasts:
            if fcst.item_id.split('_dim')[0] == item_id:
                forecast_list.append(fcst)
        
        
        figures.append(plot_forecast_multivar(
            input_list,
            label_list,
            forecast_list,
            data,
            item_id,
            dataset_label=dataset_label,
            show_plot=show_plot,
        ))
    return figures
            
def plot_forecast_multivar(
    input_list, label_list, forecast_list, data, item_id, dataset_label=None, show_plot=True
):
    n_variates = len(input_list)
    fig, axes = plt.subplots(n_variates, 1, figsize=(4 * n_variates, 4), sharex=True)
    if n_variates == 1:
        axes = [axes]
    context_means = np.zeros(n_variates)
    for var, (input, label, fcst) in enumerate(zip(input_list, label_list, forecast_list)):
        for lbl, target in zip(["input", "label"], [input, label]):
            i = 0 if lbl == "input" else 1
            ax = axes[var]
            ax.tick_params(axis="x", color=DARK_GREY, labelcolor=DARK_GREY)
            ax.tick_params(axis="y", color=DARK_GREY, labelcolor=DARK_GREY)
            ax.yaxis.set_label_position("right")
            item = data[i]['start']
            if lbl == 'label':
                print(f"Var {var}: start: {item}")
            dates = pd.date_range(
                start=item.start_time,
                periods=len(target),
                freq=item.freqstr,
            )
            ax.plot(
                dates,
                target,
                color=BLUE if lbl == "input" else DARK_BLUE,
                label=f"{lbl}",
            )
            
            if lbl == "input":    
                context_means[var] = np.nanmean(target)
            
        median_forecast = fcst.median
        forecast_dates = pd.date_range(
            start=fcst.start_date.to_timestamp(),
            periods=len(median_forecast),
            freq=fcst.start_date.freqstr,
        )
        ax.plot(
            forecast_dates,
            median_forecast,
            color=PURPLE,
            linestyle="--",
            label="forecast median",
        )

        alpha = 0.05
        lower = fcst.quantile(alpha)
        upper = fcst.quantile(1 - alpha)
        ax.fill_between(
            forecast_dates,
            lower,
            upper,
            color=LIGHT_PURPLE,
            alpha=0.8,
        )

        context_mean = context_means[var]
        if not np.isnan(context_mean):
            ax.plot(
                forecast_dates,
                np.full_like(median_forecast, context_mean, dtype=float),
                color="red",
                linestyle=":",
                linewidth=1,
                label="context mean",
            )

        title_dataset = dataset_label or item_id

        for ax in axes:
            ax.grid(which="both")

        for var, ax in enumerate(axes):
            ax.set_title(f"{title_dataset}/var_{var}")
            ax.legend(loc="upper left")
    fig.tight_layout()
    if show_plot:
        fig.tight_layout()
        plt.show()
    else:
        plt.close(fig)
    return fig

def find_num_variates(dataset) -> int:
    max_dim = 0
    suffix_found = False
    for entry in iter(dataset):
        target = entry[0]["target"]
        if target.ndim != 1:
            return target.shape[0]
        item_id = entry[0]["item_id"]
        if "_dim" in item_id:
            try:
                dim_idx = int(item_id.split("_dim")[-1])
            except ValueError:
                continue
            max_dim = max(max_dim, dim_idx)
            suffix_found = True
    return max_dim + 1 if suffix_found else 1


def finetuning_curves(output_dir, experiment="n/a", figsize=(8, 6), save_path=None, max_steps=None):
    # Find the checkpoint-* folder inside the output_dir
    checkpoint_dirs = glob(os.path.join(output_dir, "checkpoint-*"))

    if not checkpoint_dirs:
        raise FileNotFoundError(f"No checkpoint-* folder found in {output_dir}")

    # Use the first checkpoint folder (or modify if you need a specific one)
    checkpoint_dir = checkpoint_dirs[-1]

    # Define the path to the trainer_state.json file
    trainer_state_path = os.path.join(checkpoint_dir, "trainer_state.json")

    # Check if the file exists
    if not os.path.exists(trainer_state_path):
        raise FileNotFoundError(f"trainer_state.json not found in {checkpoint_dir}")

    # Read the JSON file
    with open(trainer_state_path, "r") as file:
        trainer_state = json.load(file)

    # Print the contents of the JSON file
    # print(json.dumps(trainer_state, indent=4))
    # print(json.dumps(trainer_state['log_history'], indent=2))

    # plot losses
    log_history = trainer_state['log_history']
    # Extract steps, loss, and eval_loss
    steps = []
    loss = []
    eval_loss = []

    for entry in log_history:
        if "loss" in entry:
            steps.append(entry["step"])
            loss.append(entry["loss"])
        if "eval_loss" in entry:
            eval_loss.append(entry["eval_loss"])

    if max_steps is not None:
        loss = loss[:max_steps]
        eval_loss = eval_loss[:max_steps]

    # # Plot the loss gradients
    # plt.figure(figsize=figsize)
    # plt.plot(steps[1:len(loss)], np.log(np.abs(np.diff(loss))), label="Training Loss Gradient", color='navy')
    # plt.plot(steps[1:len(eval_loss)], np.log(np.abs(np.diff(eval_loss))), label="Validation Loss Gradient", color='red')
    # plt.xlabel("Steps")
    # plt.ylabel("|LogLoss| Changes")
    # plt.title(f"{experiment} Experiment - Loss Gradient")
    # plt.legend()
    # plt.grid()
    # plt.show()

    # Plot the losses
    plt.figure(figsize=figsize)
    plt.plot(steps[:len(loss)], np.log(loss), label="Training Loss", color='navy')
    plt.plot(steps[:len(eval_loss)], np.log(eval_loss), label="Validation Loss", color='red')
    plt.xlabel("Steps")
    plt.ylabel("Log-Loss")
    plt.title(f"{experiment} Experiment")
    plt.legend()
    plt.grid()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    
    
def polished_grid_plot(
    output_data: dict, colors: dict, shade: bool = True, 
    selected_data_ids: dict | None = None, figsize=(15, 5),
    caption_df: pd.DataFrame | None = None, num_fig_cols: int = 4,
    bbox_to_anchor=(1.06, 1), save_path: str | None = None,
    flatten = True, legend_ncol=5,
):
    ABBREVIATIONS = {
        "PathVolatilityRatio": "PVR",
        "MeanAttractionRatio": "MAR",
        "MeanBiasRatio": "MBR",
        "PersistenceSkillRatio": "PSR",
        "MeanSkillRatio": "MSR",
        "MSE": "MSE",
        "MAE": "MAE",
        "MAPE": "MAPE",
        "sMAPE": "sMAPE",
        "MASE": "MASE",
        "MSIS": "MSIS",
        "quantile_loss": "QL",
        "ND": "ND",
        "NormalizedMeanBias": "NMB",
        "RelativeUncertainty": "RU"
    }
    if not flatten:
        num_fig_cols = 3
    var_str = "Multivariate"
    sample_model_name = list(output_data[var_str].keys())[0]
    sample_dataset_name = list(output_data[var_str][sample_model_name].keys())[0]
    sample_variate_name = sample_dataset_name.split('/')[0]
    num_datasets = len(output_data[var_str][sample_model_name])
    print(f"Number of datasets to plot: {num_datasets}")
    grid_fig, grid_axes = plt.subplots(int(math.ceil(num_datasets/num_fig_cols)), num_fig_cols, figsize=figsize)
    if flatten:
        # flatten grid axes for easy indexing
        grid_axes = grid_axes.flatten()
        dataset_list = output_data[var_str][sample_model_name].keys()
    else:
        dataset_list = [dataset_n.split('/')[-1] for dataset_n in output_data[var_str][sample_model_name].keys() if sample_variate_name in dataset_n]
        
    dataset_indices = {key: i for i, key in enumerate(dataset_list)}

    for multivar_idx, (multivariate_str, var_outputs) in enumerate(output_data.items()):
        for model_idx, (model_name, model_outputs) in enumerate(var_outputs.items()):
            for dataset_name, list_output in model_outputs.items():
                if selected_data_ids and selected_data_ids.get(dataset_name):
                    data_output_key = selected_data_ids[dataset_name]
                else:
                    data_output_key = np.random.choice(list(list_output.keys())) # randomly select one dataset to plot for each model
                    selected_data_ids[dataset_name] = data_output_key
                    print(f"{dataset_name}: {list_output[data_output_key]['item_id']} selected.")
                    
                data_output = list_output[data_output_key]
                
                # data_output keys: 'dataset_label', 'input_list', 'label_list', 'forecast_list', 'data', 'item_id'
                _dataset_name, input_list, label_list, forecast_list, data, item_id = data_output.values()
                assert dataset_name == _dataset_name, "Dataset name mismatch!"
                data_idx = dataset_indices[dataset_name if flatten else dataset_name.split('/')[-1]]
                
                if flatten:
                    # set ax to the next cell 
                    ax = grid_axes[data_idx]
                else:
                    if data_idx == 0:
                        grid_axes[data_idx, 0].set_title("Univariate")
                        grid_axes[data_idx, 1].set_title("Multivariate")
                        grid_axes[data_idx, 2].set_title("Leading Covariate")
                        for j in range(3):
                            # make title bold   
                            grid_axes[data_idx, j].title.set_fontweight('bold')
                            grid_axes[data_idx, j].title.set_fontsize(12)
                    ax = grid_axes[data_idx, 2] if 'multivariate_lagged' in dataset_name else grid_axes[data_idx, 1] if 'multivariate' in dataset_name else grid_axes[data_idx, 0]
                if caption_df is not None:
                    eval_metric_cols = [col for col in caption_df.columns if ('eval_metrics' in col[0] and '%Improv' in col[1])]
                    in_caption_df = caption_df[caption_df['dataset'].str.contains(dataset_name)]
                    caption = 'Improv ' + ', '.join([f"{ABBREVIATIONS[col[0].replace('eval_metrics/', '').replace('[0.5]', '').replace('[512]', '')]}: {in_caption_df[col].values[0]:.1f}%" for col in eval_metric_cols])
                    if len(caption) > 0:
                        ax.set_xlabel(caption)
                
                # plot_one_axis(ax, input_list, label_list, forecast_list, data, colors, shade=shade)
                for var, (input, label, fcst) in enumerate(zip(input_list, label_list, forecast_list)):
                    if model_idx == 0:
                        for lbl, target in zip(["input", "label"], [input, label]):
                            if var !=0 and lbl == "label": # only plot input for the input to avoid clutter
                                continue
                            k = 0 if lbl == "input" else 1
                            dates = pd.date_range(
                                start=data[k]["start"].to_timestamp(), periods=target.size, freq=data[k]["freq"],
                            )
                            if lbl == "input":
                                ax.axvline(x=dates[-1], color=colors["boundary"], linestyle="--", linewidth=1)
                            ax.plot(
                                dates, target, 
                                color=colors["input"] if var == 0 else colors["leading_covariate"], 
                                label=(f"Target Variable" if var == 0 else "Leading Covariate") if (lbl == "input" and data_idx == 0 and multivariate_str == "Multivariate") else None,
                                # linewidth = 3 if var == 0 else 2
                            )
                    if var == 0: # only plot forecast for the first variable to avoid clutter
                        # if model_idx != 0:
                        #     continue
                        median_forecast = fcst.median
                        forecast_dates = pd.date_range(
                            start=fcst.start_date.to_timestamp(), periods=len(median_forecast), freq=fcst.start_date.freqstr,
                        )
                        ax.plot(
                            forecast_dates, median_forecast, color=colors[model_name], #linewidth=2,
                            label=f"{model_name}" if (data_idx == 0 and multivar_idx == 0) else None,
                        )
                        if shade:
                            alpha = 0.05
                            lower = fcst.quantile(alpha)
                            upper = fcst.quantile(1 - alpha)
                            ax.fill_between(forecast_dates, lower, upper, color=colors[model_name], alpha=0.2)
                # remove x ticks
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_ylabel(dataset_name if flatten else dataset_name.split("/")[-1]) # 
                
                # sort the legend in this order: "Target Variable", "Leading Covariate", "Model A", "Model B"
                desired_order = ["Target Variable", "Leading Covariate"]

                # Extract handles and labels from all axes
                axes_handles_labels = [ax.get_legend_handles_labels() for ax in grid_axes.flatten()]
                handles_labels = [hl for hl in axes_handles_labels if hl[1]]  # filter out axes with no labels
                handles, labels = zip(*handles_labels)

                # Flatten handles and labels
                handles = [h for sublist in handles for h in sublist]
                labels = [l for sublist in labels for l in sublist]

                # Create a mapping of labels to handles
                label_handle_map = dict(zip(labels, handles))

                # Sort handles and labels based on the desired order, with remaining labels in alphabetical order
                sorted_handles = [label_handle_map[label] for label in desired_order if label in label_handle_map]
                sorted_labels = [label for label in desired_order if label in label_handle_map]

                # Add remaining labels in alphabetical order
                remaining_labels = sorted(set(labels) - set(desired_order))
                sorted_handles.extend([label_handle_map[label] for label in remaining_labels])
                sorted_labels.extend(remaining_labels)
    


    # Add one legend for the entire figure outside the grid
    grid_fig.legend(sorted_handles, sorted_labels,loc='upper center', bbox_to_anchor=bbox_to_anchor, ncol=legend_ncol, frameon=True)

    grid_fig.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight') 
    plt.show()