
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import yfinance

import matplotlib.pyplot as plt
import torch

import DataDownload # Download stock data from Yahho Finance
import LinearModel



class model_training():
    def __init__(self,model,tick,start,end,interval,train_ratio,max_lags,no_epochs,lr,
        rtol = 1e-6,trade_signal = "sign",threshold = 0.001):
        self.model = model
        self.tick = tick
        self.start = start
        self.end = end
        self.interval = interval
        self.train_ratio = train_ratio
        self.max_lags = max_lags
        self.trade_signal = trade_signal
        self.threshold = threshold


        # nn parameters
        self.LossFunction = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.no_epochs = no_epochs
        self.lr = lr
        self.rtol = rtol
        self.model_is_trained = False

        self.df = self.get_finance_data()
        self.prepare_lag_test_data_tensors()
        # Acturally train the model specified
        self.train_model()
        self.get_trade_results(self.trade_signal, self.threshold)


    def get_finance_data(self):
        from DataDownload import download_one_data
        df = download_one_data(self.tick,self.start,self.end,self.interval)
        return df

    def plot_finance_data(self,column,display_percentage = True):
        df = self.df
        if display_percentage:
            plt.plot(pd.to_datetime(df.index),np.exp(df[column])*100 - 1,label=column)
            plt.ylabel('Percentage Change')
        else:
            plt.plot(pd.to_datetime(df.index),df[column],label=column)
        plt.legend()
        plt.grid()

    def prepare_lag_test_data_tensors(self):
        def features_lag_(data: pd.DataFrame, lag):
            df = pd.DataFrame(index = data.index.copy())
            df["close_log_return"] = data["close_log_return"]
            df["close_log_return_cum"] = data["close_log_return"].cumsum()
            if lag == 0:
                raise ValueError("lag must be greater than 0")
            if type(lag) != int:
                raise TypeError("lag must int")

            i = 1
            while i < lag + 1 :
                name = 'close_log_return_lag_' + str(i)
                df[name] = data["close_log_return"].shift(i)
                i += 1

            return df
        def get_train_test_features(data, lag, train_ratio):
            features = ['close_log_return_lag_' + str(i) for i in range(1, lag + 1)]
            target = ['close_log_return']
            df_lag = features_lag_(data, lag).dropna()
            train_idx = int(len(df_lag)*train_ratio)
            df_train, df_test = df_lag[:train_idx], df_lag[train_idx:]
            # Features
            F_train = torch.tensor(df_train[features].to_numpy(),dtype = torch.float32)
            F_test = torch.tensor(df_test[features].to_numpy(),dtype = torch.float32)

            # Targets
            T_train = torch.tensor(df_train[target].to_numpy(),dtype = torch.float32)
            T_test = torch.tensor(df_test[target].to_numpy(),dtype = torch.float32)
            return F_train, F_test, T_train, T_test
        # Get all the test and train tensors
        F_train, F_test, T_train, T_test = get_train_test_features(self.df, self.max_lags, self.train_ratio)
        self.F_train, self.F_test, self.T_train, self.T_test = F_train, F_test, T_train, T_test

    def train_model(self):
        # Features is a list of names
        # Which is the same as the maximun numbers of lags
        model = self.model
        F_train, F_test, T_train, T_test = self.F_train, self.F_test, self.T_train, self.T_test
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        print("\nTraining model...")
        train_loss_log = 0
        for epoch in range(self.no_epochs):
            # forward prediction
            T_hat = model(F_train)
            loss = self.LossFunction(T_hat, T_train)

            # backward pass
            optimizer.zero_grad()   # drop old gradient
            loss.backward()         # compute new gradient
            optimizer.step()        # update weights

            train_loss = loss.item()

            rel_loss = np.abs(train_loss-train_loss_log)/(np.abs(train_loss))

            if rel_loss < self.rtol:
                print(f"Relative Tolance Reached at Epoch {epoch}")
                break

            train_loss_log = train_loss
            if epoch % 500 == 0:
                print(f"Epoch {epoch}/{self.no_epochs}, Loss = {train_loss:.9f}")

        print("\n Training Finished!", f"\n Loss = {train_loss:.9f}")
        self.model_is_trained = True

        with torch.no_grad():
            T_hat_test = model(F_test)
            test_loss = self.LossFunction(T_hat_test, T_test).item()
            print(f"Test Loss = {test_loss:.9f}")

    def get_trade_results(self,trade_signal = "sign",threshold = 0.001):
        """Takes the trained model, and add a trade_results entery
            pd.DataFrame to self
        """
        if not self.model_is_trained:
            raise ValueError("Model is not trained yet")
        # Evaluate the model
        self.model.eval()
        F_train, F_test, T_train, T_test = self.F_train, self.F_test, self.T_train, self.T_test

        with torch.no_grad():
            T_train_hat = self.model(F_train)
            T_hat = self.model(F_test)
            test_loss = self.LossFunction(T_hat, T_test)
            train_loss = self.LossFunction(T_train_hat, T_train)
            print(f"\nTest Loss = {test_loss.item():.9f}",
                f"\nTrain Loss = {train_loss.item():.9f}")
        ##############
        ##############
        # Get predictions as numpy arrays
        t_hat = T_hat.squeeze().numpy()
        t_test = T_test.squeeze().numpy() # the reference samples

        # df = self.get_trade_results(t_hat, t_test)
        df = pd.DataFrame({
            "T_hat": t_hat,
            "T_test": t_test,

            })
        if trade_signal == "sign":
            df["signal"] = np.sign(t_hat)
        elif trade_signal == "threshold":
            df["signal"] = np.where(t_hat > threshold, 1, -1)
        df["is_won"] = (df["signal"] * t_test) > 0
        df["trade_log_return"] = t_test * df["signal"]
        df['trade_log_return_cum'] = df['trade_log_return'].cumsum()
        # df['drawdown_log'] = df['trade_log_return'] - df['trade_log_return'].cummax()

        df["equity_log"] = df["trade_log_return"].cumsum()
        df["peak_log"] = df["equity_log"].cummax().clip(lower=0)
        df["drawdown_log"] = df["equity_log"] - df["peak_log"]
        df["drawdown_percentage"] = np.expm1(df["drawdown_log"])
        self.trade_results = df

    def get_model_performance(self):

        #####################
        # Get model params
        #####################
        df = self.trade_results
        max_drawdown_log = df['drawdown_log'].min()
        drawdown_percentage = np.exp(max_drawdown_log) - 1
        win_rate = df['is_won'].mean()
        average_win = df[df['is_won']==True]['trade_log_return'].mean()
        average_loss = df[df['is_won'] == False]['trade_log_return'].mean()
        ev = win_rate * average_win + (1-win_rate) * average_loss
        total_log_return = df['trade_log_return'].sum()
        compund_return = np.exp(total_log_return)

        equity_trough = df['trade_log_return_cum'].min()
        equity_peak = df['trade_log_return_cum'].max()

        std = df['trade_log_return'].std()

        sharpe = ev/std * np.sqrt(252) # Sharp for stocks
        sharpe_crypto = ev/std * np.sqrt(365) # Sharpe for cryptos


        print(
            f"\nMaximum drawdown (log): {max_drawdown_log:.6f}"
            f"\nDrawdown percentage: {drawdown_percentage:.2%}"
            f"\nWin rate: {win_rate:.2%}"
            f"\nAverage win: {average_win:.6f}"
            f"\nAverage loss: {average_loss:.6f}"
            f"\nExpected value per trade: {ev:.6f}"
            f"\nTotal log return: {total_log_return:.6f}"
            f"\nCompounded return: {compund_return - 1:.2%}"
            f"\nEquity multiplier: {compund_return:.4f}x"
            f"\nMinimum trade log return: {equity_trough:.6f}"
            f"\nMaximum trade log return: {equity_peak:.6f}"
            f"\nStandard deviation: {std:.6f}"
            f"\nAnnualized Sharpe (stocks): {sharpe:.4f}"
            f"\nAnnualized Sharpe (crypto): {sharpe_crypto:.4f}"
        )

    def model_against_stock(self,figsize=(10,6)):
        df = self.trade_results # The trade results
        time = df.index

        close_log_return = self.T_test.squeeze().numpy().cumsum()
        return_equity_log = df['equity_log']
        # close_log_return_cum = self.df["close_log_return"].cumsum()

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(time, np.expm1(return_equity_log), label='Model',c='red')
        ax.plot(time, np.expm1(close_log_return), label='Stock',c='black')
        ax.set_title('Model vs Stock')
        ax.set_xlabel('Time')
        ax.set_ylabel('Percentage change')
        ax.grid()
        ax.legend()
        plt.show()
