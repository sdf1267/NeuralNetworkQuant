import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
def make_lags(returns, lookback):
    df = pd.DataFrame({"target": returns})
    for lag in range(1, lookback + 1):
        df[f"lag_{lag}"] = returns.shift(lag)
    return df.dropna()
def walk_forward(
    returns,
    lookback=16,
    initial_train_size=500,
    test_size=100,
):
    data = make_lags(returns, lookback)
    features = [
        f"lag_{lag}"
        for lag in range(1, lookback + 1)
    ]
    fold_results = []
    train_end = initial_train_size
    fold = 0
    while train_end + test_size <= len(data):
        train = data.iloc[:train_end]
        test = data.iloc[
            train_end:train_end + test_size
        ]
        X_train = train[features]
        y_train = train["target"]
        X_test = test[features]
        y_test = test["target"]
        # Fit scaling on training data only.
        mean = X_train.mean()
        std = X_train.std().clip(lower=1e-8)
        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std
        # Train a fresh model for this fold.
        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        result = pd.DataFrame(
            {
                "fold": fold,
                "forecast_score": predictions,
                "realized_return": y_test.to_numpy(),
            },
            index=test.index,
        )
        fold_results.append(result)
        print(
            f"Fold {fold}: "
            f"train={train.index[0]} to {train.index[-1]}, "
            f"test={test.index[0]} to {test.index[-1]}"
        )
        # Expand the training set and move forward.
        train_end += test_size
        fold += 1
    return pd.concat(fold_results)
