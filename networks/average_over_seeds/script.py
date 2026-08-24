import sys
sys.path.insert(0, '..')
import random
import forward_validation as FV
import pandas as pd 
from importlib import reload
import MLPModel, LinearModel, Model_Training, DataDownload
import numpy as np

reload(MLPModel)
reload(LinearModel)
reload(Model_Training)
reload(FV)

import torch

def evaluate_seeds(
    model_factory,
    model_name,
    seeds,
):
    results = []

    for seed in seeds:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        trained = Model_Training.model_training(
            model=model_factory(max_lags),
            tick=ticker,
            start=start,
            end=end,
            interval=interval,
            train_ratio=train_ratio,
            max_lags=max_lags,
            no_epochs=no_epoches,
            lr=lr,
            transaction_cost_bpts=tx_fees_bpts,
            weight_decay=weight_decay,
            trade_signal="threshold",
            threshold=threshold,
            verbose=False,
        )

        performance = trained.get_model_performance(
            trained.trade_results
        ).copy()

        performance["seed"] = seed
        performance["model"] = model_name
        results.append(performance)

    return pd.concat(
        results,
        ignore_index=True,
    )