import sys
sys.path.insert(0, '../../..')
import random
import pandas as pd 
from importlib import reload
import MLPModel, LinearModel, Model_Training, DataDownload
import numpy as np

reload(MLPModel)
reload(LinearModel)
reload(Model_Training)

import torch
from matplotlib import pyplot as plt
import ipywidgets as widgets
import matplotlib.pyplot as plt
def save_csv(lag_results,data):
    from pathlib import Path

    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)

    all_metrics = []
    all_paths = []
    all_bands = []

    for model_name, model_results in (
        lag_results.items()
    ):
        for lag, result in model_results.items():
            # Metrics
            metrics = result["metrics"].copy()
            metrics["model"] = model_name
            metrics["lag"] = lag
            all_metrics.append(metrics)

            # Individual seed paths: wide → long format
            paths = (
                result["paths"]
                .rename_axis("date")
                .reset_index()
                .melt(
                    id_vars="date",
                    var_name="seed_run",
                    value_name="cumulative_return",
                )
            )

            paths["model"] = model_name
            paths["lag"] = lag
            all_paths.append(paths)

            # Mean and sigma bands
            bands = (
                result["bands"]
                .rename_axis("date")
                .reset_index()
            )

            bands["model"] = model_name
            bands["lag"] = lag
            all_bands.append(bands)

    metrics_csv = pd.concat(
        all_metrics,
        ignore_index=True,
    )

    paths_csv = pd.concat(
        all_paths,
        ignore_index=True,
    )

    bands_csv = pd.concat(
        all_bands,
        ignore_index=True,
    )

    metrics_csv.to_csv(
        output_dir / "seed_metrics.csv",
        index=False,
    )

    paths_csv.to_csv(
        output_dir / "seed_paths.csv",
        index=False,
    )

    bands_csv.to_csv(
        output_dir / "seed_bands.csv",
        index=False,
    )
    cumulative_log_return = (
        data["close_log_return"]
        .cumsum()
        .rename("cumulative_log_return")
    )

    cumulative_log_return.to_csv(
        output_dir / "cumulative_log_return.csv",
        index=True,
    )

def plot_seed_bands(
    ax,
    bands,
    title,
    color,
):
    ax.plot(
        bands.index,
        bands["mean"],
        color=color,
        label="Mean",
    )

    ax.fill_between(
        bands.index,
        bands["lower_2sigma"],
        bands["upper_2sigma"],
        color=color,
        alpha=0.10,
        label=r"$\pm2\sigma$",
    )

    ax.fill_between(
        bands.index,
        bands["lower_1sigma"],
        bands["upper_1sigma"],
        color=color,
        alpha=0.25,
        label=r"$\pm1\sigma$",
    )

    ax.axhline(
        0,
        color="black",
        linewidth=1,
    )

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.grid(True)
    ax.legend()


def main_seed_average(ticker,start,end,interval,train_ratio,
                       no_epoches,lr,threshold,weight_decay = 5e-3,
                       max_lags=32,tx_fees_bpts=10,seed = 22,trade_signal = "sign",verbose=False,
                       no_seeds = 10,lags = range(1,32,3)):

    # We need the data to plot the underlying
    data = (
    DataDownload.download_one_data(
        symbol=ticker,
        start=start,
        end=end,
        interval=interval,
    )
    .dropna()
)
    def evaluate_seed_paths(
        model_factory,
        model_name,
        seeds,
        trainer_kwargs,
        ):
        metrics_list = []
        path_list = []
        max_lags = trainer_kwargs["max_lags"]
        for run, seed in enumerate(seeds):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            trained = Model_Training.model_training(
                model=model_factory(max_lags),
                **trainer_kwargs,
            )
            metrics = trained.get_model_performance(
                trained.trade_results
            ).copy()
            metrics["seed"] = seed
            metrics["model"] = model_name
            metrics_list.append(metrics)
            cumulative_return = np.expm1(
                trained.trade_results[
                    "equity_log"
                ].to_numpy()
            ) * 100
            path = pd.Series(
                cumulative_return,
                index=trained.date_test,
                name=f"{model_name}_{run}_{seed}",
            )
            path_list.append(path)
        metrics_df = pd.concat(
            metrics_list,
            ignore_index=True,
        )
        paths_df = pd.concat(
            path_list,
            axis=1,
        )
        mean = paths_df.mean(axis=1)
        std = paths_df.std(axis=1)
        bands_df = pd.DataFrame(
            {
                "mean": mean,
                "std": std,
                "lower_1sigma": mean - std,
                "upper_1sigma": mean + std,
                "lower_2sigma": mean - 2 * std,
                "upper_2sigma": mean + 2 * std,
            }
        )
        return metrics_df, paths_df, bands_df


    random_seeds = [random.randint(0,1_000_000) for _ in range(no_seeds)]  # pick 500 random seeds

    # seed_MLP = evaluate_seeds(MLPModel.MLP,model_name = "MLP",seeds = random_seeds)
    # seed_LIN = evaluate_seeds(LinearModel.LinearModel,model_name = "Linear",seeds = random_seeds) 
    trainer_kwargs = {
        "tick": ticker,
        "start": start,
        "end": end,
        "interval": interval,
        "train_ratio": train_ratio,
        "max_lags": max_lags,
        "no_epochs": no_epoches,
        "lr": lr,
        "transaction_cost_bpts": tx_fees_bpts,
        "weight_decay": weight_decay,
        "trade_signal": trade_signal,
        "threshold": threshold,
        "verbose": False,
        "data": data
    }

   

    lag_results = {
        "MLP": {},
        "Linear": {},
    }

    model_factories = {
        "MLP": MLPModel.MLP,
        "Linear": LinearModel.LinearModel,
    }

    for lag in lags:
        print(f"\nRunning lag {lag}...")

        lag_kwargs = trainer_kwargs.copy()
        lag_kwargs["max_lags"] = lag
 

        for model_name, model_factory in (
            model_factories.items()
        ):
            print(f"  {model_name}")

            metrics, paths, bands = (
                evaluate_seed_paths(
                    model_factory=model_factory,
                    model_name=model_name,
                    seeds=random_seeds,
                    trainer_kwargs=lag_kwargs,
                )
            )

            lag_results[model_name][lag] = {
                "metrics": metrics,
                "paths": paths,
                "bands": bands,
            }



    # data = 

    def plot_selected_lag(lag):
        mlp_bands = lag_results[
            "MLP"
        ][lag]["bands"]

        lin_bands = lag_results[
            "Linear"
        ][lag]["bands"]

        # Underlying return aligned to each test period.
        underlying_returns = data[
            "close_log_return"
        ]

        if isinstance(
            underlying_returns,
            pd.DataFrame,
        ):
            underlying_returns = (
                underlying_returns.squeeze(
                    "columns"
                )
            )

        underlying_returns = (
            underlying_returns
            .reindex(mlp_bands.index)
            .fillna(0.0)
        )

        underlying_cumulative = (
            np.expm1(
                underlying_returns.cumsum()
            )
            * 100
        )

        fig, (ax1, ax2) = plt.subplots(
            2,
            1,
            figsize=(11, 9),
            sharex=True,
            sharey=True,
        )
        fig.suptitle(ticker)

        plot_seed_bands(
            ax1,
            mlp_bands,
            title=f"MLP — lookback {lag}",
            color="red",
        )

        plot_seed_bands(
            ax2,
            lin_bands,
            title=f"Linear model — lookback {lag}",
            color="blue",
        )

        for ax in (ax1, ax2):
            ax.plot(
                underlying_cumulative.index,
                underlying_cumulative,
                color="black",
                linewidth=1.5,
                label="Underlying",
            )

            ax.set_ylabel(
                "Cumulative return (%)"
            )

            ax.legend()

        ax2.set_xlabel("Date")

        fig.tight_layout()
        plt.show()

    save_csv(lag_results,data)
    widgets.interact(
        plot_selected_lag,
        lag=widgets.SelectionSlider(
            options=sorted(lags),
            value=sorted(lags)[0],
            description="Lookback:",
            continuous_update=False,
        ),
    )

# def main_benchmark

    