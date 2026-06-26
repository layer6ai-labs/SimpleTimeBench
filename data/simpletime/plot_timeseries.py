import matplotlib.pyplot as plt
import pandas as pd


def plot_multivariate_timeseries_subplots(
    list_of_dfs, nplt=None, nplt_max=10, title=None
):
    """
    Plots a multivariate time series dataset with each dataframe in its own subplot.

    Each column within a dataframe is treated as a separate variate and plotted
    on the same subplot.

    Args:
        list_of_dfs (list): A list of pandas dataframes.
        nplt (int): number of samples to plot
    """

    n_dfs = len(list_of_dfs)
    if nplt is None:
        nplt = n_dfs
    if nplt > nplt_max:
        print(
            f"Warning, nplt={nplt} > nplt_max={nplt_max}.\nIncrease max if you really want to plot that many in one subplot!"
        )

    fig, axes = plt.subplots(
        nrows=nplt, ncols=1, figsize=(10, 4 * nplt), sharex=True, squeeze=False
    )
    fig.suptitle(title if title else "Time Series Samples", fontsize=16)
    # Ensure axes is an array even for a single subplot
    if n_dfs == 1:
        axes = [axes]

    for i, df in enumerate(list_of_dfs[:nplt]):

        is_series = isinstance(df, pd.Series)
        is_dataframe = isinstance(df, pd.DataFrame)

        ax = axes[i, 0]

        if is_series:
            ax.plot(df.index, df.values, label="Value")

        # Plot each column of the dataframe on the subplot
        if is_dataframe:
            for column in df.columns:
                ax.plot(df.index, df[column], label=f"{column}")

        ax.set_title(f"DataFrame {i+1}")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True)

    # Set a common xlabel for all subplots
    axes[-1, 0].set_xlabel("Time")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    from timeseries_generator import TimeSeriesGenerator

    generator = TimeSeriesGenerator(random_state=13579)

    multivariate_dataset = generator.create_multivariate_dataset(
        num_samples=3,
        length=100,
        num_series=3,
        lagged=True,
        lag_amount=5,
        distributions="random_walk",
    )

    # Call the function to plot the data
    plot_multivariate_timeseries_subplots(multivariate_dataset)
