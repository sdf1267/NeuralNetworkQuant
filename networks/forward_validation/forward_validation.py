import random
import numpy as np
import pandas as pd
import torch


import DataDownload
import MLPModel
import LinearModel

def make_lagged_data(returns, lookback):
    data = pd.DataFrame(
        {
            "target": returns # this is usually closed log return
        }
    )
    for lag in range(1,lookback+1): # run until lookback
        name = "lag_" + str(lag)
        data[name] = data["target"].shift(lag)
    return data.dropna()

def train_one_fold(
    model,train_df,test_df,features,
    epochs = 10_000,
    lr=1e-3,
    weight_decay = 0.0,
    rtol = 1e-7, atol = 0
):

    F_train = torch.tensor(
        train_df[features].to_numpy(),
        dtype=torch.float32,
    )
    F_test = torch.tensor(
        test_df[features].to_numpy(),
        dtype=torch.float32,
    )
    T_train = torch.tensor(
        train_df["target"].to_numpy(),
        dtype=torch.float32,
    ).reshape(-1, 1)

    # The imput features are too small, rescale them to better fit the data
    F_mean = F_train.mean(dim = 0, keepdim = True)
    F_std = F_train.std(dim=0, keepdim = True).clamp_min(1e-8)

    T_mean = T_train.mean()
    T_std = T_train.std().clamp_min(1e-8)
    F_train_scaled = (F_train - F_mean)/F_std
    F_test_scaled = (F_test - F_mean)/F_std

    # T_test_scaled = (T_test - T_mean)/T_std
    T_train_scaled = (T_train - T_mean)/T_std

    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay = weight_decay)

    train_loss_log = 0

    model.train()
    for epoch in range(epochs):
        # forward prediction
        T_hat = model(F_train_scaled)
        loss = loss_function(T_hat, T_train_scaled)

        # backward pass
        optimizer.zero_grad()   # drop old gradient
        loss.backward()         # compute new gradient
        optimizer.step()        # update weights

        train_loss = loss.item()

        rel_loss = np.abs(train_loss-train_loss_log)/(np.abs(train_loss))
        abs_loss = np.abs(train_loss-train_loss_log)
        if rel_loss < rtol:
            print(f"Relative Tolance Reached at Epoch {epoch}")
            break

        if abs_loss < atol:
            print(f"Relative Tolance Reached at Epoch {epoch}")
            break
        train_loss_log = train_loss

    model.eval()
    with torch.no_grad():
        prediction = model(F_test_scaled).squeeze()
    return (prediction*T_std+T_mean).numpy()

def walk_forward(returns, model_factory,
    lookback=16, initial_train_size=500, test_size=100,
    epochs = 10_000,lr=1e-3, weight_decay=0.0, seed=42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    data = make_lagged_data(returns,lookback)

    features = [f"lag_{lag}" for lag in range (1, lookback + 1)]
    results = []
    train_end = initial_train_size
    fold = 0

    while train_end + test_size <= len(data): #while the fold is still smaller
        # Get the test and train data
        train_df = data.iloc[:train_end]
        test_df = data.iloc[train_end: train_end + test_size]

        # Model to be trained on
        model = model_factory( lookback )

        predictions = train_one_fold(model =model,train_df = train_df,test_df = test_df,
            features = features,
            epochs = epochs,
            lr = lr,
            weight_decay = weight_decay
        )

        fold_result = pd.DataFrame(
            {
                'fold' : fold,
                'predictions': predictions,
                'T_test': test_df["target"].to_numpy()
            }, index = test_df.index
        )

        results.append(fold_result)
        print(f'Fold {fold}:', f"train = {len(train_df)}", f"test={len(test_df)}")
        train_end += test_size
        fold += 1

    return pd.concat(results).sort_index()


def data_analysis(
    fold, transaction_fees_bpts = 10
):
    predictions = fold['predictions']
    actual = fold['T_test']
    fold['signal'] = np.sign(fold['predictions'])
    turnover = (fold['signal'] - fold['signal'].shift(1,fill_value=0)).abs()
    transaction_fees_log = np.log1p(-transaction_fees_bpts/10_000)
    tx_fees = turnover * transaction_fees_log
    
    fold['trade_log_return'] = (fold['T_test'])* fold["signal"] + tx_fees
    active = fold['signal'] !=0
    equity = fold['trade_log_return'].cumsum()
    drawdown = (equity - equity.cummax().clip(lower=0))
    
    zero_mse = np.mean(actual**2)
    model_mse = np.mean((actual-predictions)**2)
    
    metrics = pd.DataFrame([{ 
        'Observations' : len(fold),
        'pearson_ic' : predictions.corr(actual,method = 'pearson'),
        "spearman_ic": predictions.corr(actual,method = 'spearman'),
        'zero_mse': zero_mse,
        'model_mse': model_mse,
        'mse_ratio': model_mse/zero_mse if zero_mse>0 else np.nan,
        'direction_accuracy': (fold['signal'] == np.sign(actual)).mean(),
        'active_win_rate': (fold.loc[active,'trade_log_return']>0).mean(),
        'active_fraction': active.mean(),
        "total_turnver": turnover.sum(),
        'net_return': np.expm1(fold['trade_log_return'].sum()),
        'annualized_sharpe': (fold['trade_log_return'].mean()/fold['trade_log_return'].std()) * np.sqrt(252),
        'max_drawdown': np.expm1(drawdown.min())
    }])
    return metrics
    
 
