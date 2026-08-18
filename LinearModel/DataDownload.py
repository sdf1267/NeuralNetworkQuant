# %%
from pathlib import Path

# import binance
import numpy as np
import yfinance as yf
import pandas as pd



def download_data(
    symbol: str| list[str],
    start: str,
    end: str,
    interval: str,
):
    data = yf.download(symbol, start=start, end=end, interval=interval)
    for symb in symbol:

        data[("close_log_return",symb)] = np.log(data[("Close",symb)]) - np.log(data[("Close",symb)].shift(1))
        data[("close_cumulative_log_return",symb)] =  data[("close_log_return",symb)].cumsum()
        data[("open_close_log_diff",symb)] = np.log(data[("Open",symb)]) - np.log(data[("Close",symb)].shift(1))
        data[("overnight_log_gap",symb)] = np.log(data[("Open",symb)]) - np.log(data[("Open",symb)].shift(1)) # measured with the day and the previous date
    return data

def download_one_data(
    symbol: str| list[str],
    start: str,
    end: str,
    interval: str,
):
    data = yf.download(symbol, start=start, end=end, interval=interval)
    data["close_log_return"] = np.log(data[("Close",symbol)]) - np.log(data[("Close",symbol)].shift(1))
    data["close_cumulative_log_return"] =  data["close_log_return"].cumsum()
    data["open_close_log_diff"] = np.log(data[("Open",symbol)]) - np.log(data[("Close",symbol)].shift(1))
    data["overnight_log_gap"] = np.log(data[("Open",symbol)]) - np.log(data[("Open",symbol)].shift(1)) # measured with the day and the previous date
    return data

def append_lag_close_log_return(data:pd.DataFrame,  lag:int, symbol: str):
    if lag == 0:
        raise ValueError("lag must be greater than 0")
    if type(lag) != int:
        raise TypeError("lag must int")
    i = 1
    while i < lag + 1 :
        name = 'close_log_return_lag_' + str(lag)
        data[(name,symbol)] = data[("close_log_return", symbol)].shift(lag)
        i += 1
