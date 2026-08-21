
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
                 transaction_cost_bpts = 10, atol = 1e-9,
        rtol = 1e-6,trade_signal = "sign",threshold = 0.001,verbose = False):
        self.model = model
        self.tick = tick
        self.start = start
        self.end = end
        self.interval = interval
        self.train_ratio = train_ratio
        self.max_lags = max_lags
        self.trade_signal = trade_signal
        self.threshold = threshold
        self.verbose = verbose
        # Transaction costs, in base points
        self.transaction_cost_bpts = transaction_cost_bpts


        # nn parameters
        self.LossFunction = torch.nn.MSELoss()
        # self.LossFunction = (torch.nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.no_epochs = no_epochs
        self.lr = lr
        self.rtol = rtol
        self.atol = atol
        self.model_is_trained = False

        self.df = self.get_finance_data()
        self.prepare_lag_test_data_tensors()
        # Acturally train the model specified
        self.train_model()
        self.get_trade_results(self.trade_signal, self.threshold, self.transaction_cost_bpts)


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
            
            # Times for plots
            date_train = pd.to_datetime(df_train.index)
            date_test = pd.to_datetime(df_test.index)
            
            return F_train, F_test, T_train, T_test, date_train, date_test
        # Get all the test and train tensors
        F_train, F_test, T_train, T_test, date_train, date_test = get_train_test_features(self.df, self.max_lags, self.train_ratio)
        self.F_train, self.F_test, self.T_train, self.T_test, self.date_train, self.date_test = F_train, F_test, T_train, T_test,date_train, date_test

    def train_model(self):
        # Features is a list of names
        # Which is the same as the maximun numbers of lags
        model = self.model
        F_train, F_test, T_train, T_test = self.F_train, self.F_test, self.T_train, self.T_test
        
        # The imput features are too small, rescale them to better fit the data 
        F_mean = F_train.mean(dim=0, keepdim=True)
        F_std = F_train.std(dim=0, keepdim=True).clamp_min(1e-8)

        T_mean = T_train.mean(dim=0, keepdim=True)
        T_std = T_train.std(dim=0, keepdim=True).clamp_min(1e-8)
        
        
        self.T_mean = T_mean 
        self.T_std = T_std
        self.F_mean = F_mean
        self.F_std = F_std
        
        F_train_scaled = (F_train - F_mean)/F_std 
        F_test_scaled = (F_test - F_mean)/F_std 
        
        T_test_scaled = (T_test - T_mean)/T_std
        T_train_scaled = (T_train - T_mean)/T_std
        
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        if self.verbose:
            print("\nTraining model...")
        train_loss_log = 0
        for epoch in range(self.no_epochs):
            # forward prediction
            T_hat = model(F_train_scaled)
            loss = self.LossFunction(T_hat, T_train_scaled)

            # backward pass
            optimizer.zero_grad()   # drop old gradient
            loss.backward()         # compute new gradient
            optimizer.step()        # update weights

            train_loss = loss.item()

            rel_loss = np.abs(train_loss-train_loss_log)/(np.abs(train_loss))
            abs_loss = np.abs(train_loss-train_loss_log)
            if rel_loss < self.rtol:
                if self.verbose: 
                    print(f"Relative Tolance Reached at Epoch {epoch}")
                break
            if abs_loss < self.atol:
                if self.verbose: 
                    print(f"Relative Tolance Reached at Epoch {epoch}")
                break

            train_loss_log = train_loss
            if epoch % 500 == 0 and self.verbose:
                print(f"Epoch {epoch}/{self.no_epochs}, Loss = {train_loss:.9f}")
        if self.verbose:
            print("\n Training Finished!", f"\n Loss = {train_loss:.9f}")
        self.model_is_trained = True

        with torch.no_grad():
            T_hat_test = model(F_test_scaled)
            test_loss = self.LossFunction(T_hat_test, T_test_scaled).item()
            if  self.verbose:
                print(f"Test Loss = {test_loss:.9f}")   
                
    def get_MR_results(self,transaction_fees_bpts = 0):
        '''Get the trading results of a simple mean reversion (MR) stragety for comparism'''
        if not self.model_is_trained:
            raise ValueError("Model is not trained yet")
        # Evaluate the model
        self.model.eval()
        F_train, F_test, T_train, T_test = self.F_train, self.F_test, self.T_train, self.T_test
        F_train_scaled = (F_train - self.F_mean)/self.F_std 
        F_test_scaled = (F_test - self.F_mean)/self.F_std 
        with torch.no_grad():
            T_train_hat = self.model(F_train_scaled) * self.T_std + self.T_mean
            T_hat = self.model(F_test_scaled) * self.T_std + self.T_mean
            test_loss = self.LossFunction(T_hat, T_test)
            train_loss = self.LossFunction(T_train_hat, T_train)
            if self.verbose:
                print(f"\nTest Loss = {test_loss.item():.9f}",
                    f"\nTrain Loss = {train_loss.item():.9f}")
        ##############
        ##############
        # Get predictions as numpy arrays
        t_hat = T_hat.squeeze().numpy()
        t_test = T_test.squeeze().numpy() # the reference samples
        
        df = pd.DataFrame({
            "T_hat": t_hat,
            "T_test": t_test,
            })
        
        df["trade_log_return"] = t_test 
        df["trade_log_return_lag_1"] = df["trade_log_return"].shift(1)
        df["trade_log_return_lag_max"] = df["trade_log_return"].shift(self.max_lags)
        # Make sure that the number of elements is same as the NN results 
        df.dropna()
        # Signals 
        df["signal"] = -np.sign(df["trade_log_return_lag_1"])
        df["turnover"] = (df["signal"] - df["signal"].shift(1,fill_value = 0)).abs()
        df["is_won"] = (df["signal"] * df["trade_log_return"] ) > 0 
        transaction_fees_log = np.log1p(-transaction_fees_bpts/10_000)
        df["tx_fees"] = df["turnover"] * transaction_fees_log
        
        df["trade_log_return"] = (t_test)* df["signal"] + df["tx_fees"] 
        df['trade_log_return_cum'] = df['trade_log_return'].cumsum()
        
        df["equity_log"] = df["trade_log_return"].cumsum()
        df["peak_log"] = df["equity_log"].cummax().clip(lower=0)
        df["drawdown_log"] = df["equity_log"] - df["peak_log"]
        df["drawdown_percentage"] = np.expm1(df["drawdown_log"])
        
        self.MR_df = df 

        return

    def get_trade_results(self, 
                          trade_signal = "sign",threshold = 0.001,
                          transaction_fees_bpts = 0):
        """Takes the trained model, and add a trade_results entery
            pd.DataFrame to self
        """
        if not self.model_is_trained:
            raise ValueError("Model is not trained yet")
        # Evaluate the model
        self.model.eval()
        F_train, F_test, T_train, T_test = self.F_train, self.F_test, self.T_train, self.T_test
        F_train_scaled = (F_train - self.F_mean)/self.F_std 
        F_test_scaled = (F_test - self.F_mean)/self.F_std 
        with torch.no_grad():
            T_train_hat = self.model(F_train_scaled) * self.T_std + self.T_mean
            T_hat = self.model(F_test_scaled) * self.T_std + self.T_mean
            test_loss = self.LossFunction(T_hat, T_test)
            train_loss = self.LossFunction(T_train_hat, T_train)
            if self.verbose:
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
        elif trade_signal == "threshold" :
            df["signal"] = np.where(t_hat > threshold, 1, -1)
        df["is_won"] = (df["signal"] * t_test) > 0
        
        # Compute turnover 
        df["turnover"] = (df["signal"] - df["signal"].shift(1,fill_value = 0)).abs()
        transaction_fees_log = np.log1p(-transaction_fees_bpts/10_000)
        df["tx_fees"] = df["turnover"] * transaction_fees_log
        
        df["trade_log_return"] = (t_test)* df["signal"] + df["tx_fees"] 
        df['trade_log_return_cum'] = df['trade_log_return'].cumsum()
        

        df["equity_log"] = df["trade_log_return"].cumsum()
        df["peak_log"] = df["equity_log"].cummax().clip(lower=0)
        df["drawdown_log"] = df["equity_log"] - df["peak_log"]
        df["drawdown_percentage"] = np.expm1(df["drawdown_log"])
        self.trade_results = df

    def get_model_performance(self,df):

        #####################
        # Get model params
        #####################
        # df = self.trade_results
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


        self.performance_df = pd.DataFrame([{
            "max_drawdown_log": max_drawdown_log,
            "drawdown_percentage": drawdown_percentage,
            "win_rate": win_rate,
            "average_win": average_win,
            "average_loss": average_loss,
            "expected_value_per_trade": ev,
            "total_log_return": total_log_return,
            "compounded_return": compund_return - 1,
            "equity_multiplier": compund_return,
            "equity_trough": equity_trough,
            "equity_peak": equity_peak,
            "return_std": std,
            "annualized_sharpe_stocks": sharpe,
            "annualized_sharpe_crypto": sharpe_crypto,
        }])

        # print(
        #     f"\nMaximum drawdown (log): {max_drawdown_log:.6f}"
        #     f"\nDrawdown percentage: {drawdown_percentage:.2%}"
        #     f"\nWin rate: {win_rate:.2%}"
        #     f"\nAverage win: {average_win:.6f}"
        #     f"\nAverage loss: {average_loss:.6f}"
        #     f"\nExpected value per trade: {ev:.6f}"
        #     f"\nTotal log return: {total_log_return:.6f}"
        #     f"\nCompounded return: {compund_return - 1:.2%}"
        #     f"\nEquity multiplier: {compund_return:.4f}x"
        #     f"\nMinimum trade log return: {equity_trough:.6f}"
        #     f"\nMaximum trade log return: {equity_peak:.6f}"
        #     f"\nStandard deviation: {std:.6f}"
        #     f"\nAnnualized Sharpe (stocks): {sharpe:.4f}"
        #     f"\nAnnualized Sharpe (crypto): {sharpe_crypto:.4f}"
        # )

        return self.performance_df

    def model_against_stock(self,ax):
        df = self.trade_results # The trade results
        

        close_log_return = self.T_test.squeeze().numpy().cumsum()
        return_equity_log = df['equity_log']
        # close_log_return_cum = self.df["close_log_return"].cumsum()
        ax.plot(self.date_test, np.expm1(return_equity_log) * 100, label='Model',c='red')
        ax.plot(self.date_test, np.expm1(close_log_return) * 100, label='Stock',c='black')
        ax.set_title('Model vs Stock')
        ax.set_xlabel('Time')
        ax.set_ylabel('Percentage change (%)')
        ax.grid()
        ax.legend()
        # plt.show()
