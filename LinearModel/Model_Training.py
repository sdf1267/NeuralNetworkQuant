
import pandas as pd
import numpy as np
import yfinance

import matplotlib.pyplot as plt
import torch

import DataDownload # Download stock data from Yahho Finance
import LinearModel # Linear NN model.


class model_training():
    def __init__(self,model,tick,start,end,interval):
        self.model = model
        self.tick = tick
        self.start = start
        self.end = end
        self.interval = interval

        self.df = self.get_finance_data()


    def get_finance_data(self):
        from DataDownload import download_data
        df = download_data(self.tick,self.start,self.end,self.interval)
        return df.droplevel(1,axis = 1)

    def plot_finance_data





def main_training(
    tickers,
    start,
    end,
    interval,
    model
):
    df = DataDownload.download_data(tickers,start,end,interval)








def __init__():
    main()
