#
# Author: Sida Tian 
# Max Planck Institute for Solid State Research
#

import sys
sys.path.insert(0, '..')


import random
import pandas as pd 
from importlib import reload
import MLPModel, LinearModel, Model_Training, DataDownload
import numpy as np

reload(MLPModel)
reload(LinearModel)
reload(Model_Training)

import torch
seed = 22

########################################

def plot_models_return(ticker,start,end,interval,train_ratio,
                       no_epoches,lr,threshold,weight_decay = 5e-3,
                       max_lags=32,tx_fees_bpts=10,seed = 22,trade_signal = "sign",verbose=False):
    """Plot the seed plots """
    
    torch.manual_seed(seed)
    # ticker = ticker
    # start = start    # start date
    # end  = end     # end date
    # interval = interval        # interval
    # train_ratio = train_ratio
    # no_epoches = no_epoches
    # lr = lr
    # tx_fees_bpts = tx_fees_bpts
    # threshold = threshold
    # trade_signal

    # weight_decay = 5e-3

    # max_lags = 32

    modelMLP = MLPModel.MLP # multiplayer perception 

    Model = Model_Training.model_training(model = modelMLP(max_lags),
                                tick = ticker,
                                start = start,
                                end = end ,
                                interval = interval,
                                train_ratio = train_ratio,
                                max_lags = max_lags,
                                no_epochs=no_epoches,
                                lr = lr,
                                rtol = 1e-5,
                                trade_signal = trade_signal,
                                threshold=threshold,
                                transaction_cost_bpts= tx_fees_bpts,
                                verbose = verbose,
                                weight_decay= weight_decay)

    modelLinear = LinearModel.LinearModel # Linear Model equivalent to AR tine series

    ModelLin = Model_Training.model_training(model = modelLinear(max_lags),
                                tick = ticker,
                                start = start,
                                end = end ,
                                interval = interval,
                                train_ratio = train_ratio,
                                max_lags = max_lags,
                                no_epochs=no_epoches,
                                lr = lr,
                                rtol = 1e-5 ,
                                trade_signal = trade_signal,
                                threshold=threshold,
                                transaction_cost_bpts= tx_fees_bpts,
                                verbose = verbose,
                                weight_decay= weight_decay)
    
    
    #################
    # Plot the Model return, rolling IC, and rolling win rate 
    fx_fees_bpts = 10
    threshold = 0.0000#0.0000001

    window = Model.rolling_window
    Model.get_MR_results()
    MR_df = Model.MR_df
    MR_cum_return = np.expm1(MR_df["trade_log_return_cum"]) * 100



    import matplotlib.pyplot as plt
    fig, (ax1,ax2,ax3) = plt.subplots(3,1, figsize = (11, 13.5))

    Model.get_trade_results(trade_signal="threshold",
                            threshold=threshold, 
                            transaction_fees_bpts= fx_fees_bpts,
                            window = window  )

    ModelLin.get_trade_results(trade_signal="threshold",
                            threshold=threshold, 
                            transaction_fees_bpts= fx_fees_bpts, 
                            window = window)

    df_MLP = Model.trade_results
    df_LIN = ModelLin.trade_results

    dates = Model.date_test
    return_MLP = np.expm1(df_MLP["trade_log_return_cum"]) * 100
    return_stock = np.expm1(Model.T_test.squeeze().numpy().cumsum()) * 100
    return_LIN = np.expm1(df_LIN["trade_log_return_cum"]) * 100

    # ax1.plot(dates,MR_cum_return,c='green',label='Mean Reversion',ls='dotted')


    ax1.plot(dates, return_stock, c='black',label="stock return")
    ax1.plot(dates, return_MLP, c='darkred',label='MLP')
    ax1.plot(dates, return_LIN, c='darkblue',label='LIN',ls='-.')

    ax1.grid()

    ax1.set_title("Return (%)")
    ax1.legend()


    # Plot rolling IC

    df = Model.trade_results

    Model.get_model_performance(df)
    rolling_ic_MLP = df["rolling_IC"].dropna()

    rolling_WR_MLP = df["is_won"].rolling(window = window, min_periods=window).mean().dropna()
    df = ModelLin.trade_results
    ModelLin.get_model_performance(df)
    rolling_ic_LIN = df["rolling_IC"].dropna()
    rolling_WR_LIN = df["is_won"].rolling(window = window, min_periods=window).mean().dropna()

    ax2.plot(Model.date_test[window-1:], rolling_ic_MLP, c='darkred', label='MLP IC')
    ax2.plot(Model.date_test[window-1:], rolling_ic_LIN, c='darkblue', label='LIN IC',ls='-.')
    ax2.grid()
    ax2.set_title("Rolling IC")
    ax2.legend()


    # Rolling winrate 
    # rolling_WR = df["is_won"].rolling(window = window, min_periods=window).mean()

    ax3.set_title("Rolling Win Rate")

    ax3.plot(dates[window-1:], rolling_WR_MLP, c='darkred',label='MLP rolling WR')
    ax3.plot(dates[window-1:], rolling_WR_LIN, c='darkblue',label='LIN rolling WR')
    ax3.legend()
    ax3.grid()

    for ax in (ax1,ax2,ax3):
        ax.set_xlim(Model.date_test[0], Model.date_test[-1])



# Model.date_test
    


        


