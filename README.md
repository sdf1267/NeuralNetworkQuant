

# Log-return forecasting: Comparing Linear and Nonlinear Neural Networks



This is a simple, exploratory research on Neural networks (NN) and their applicability to market predictions. We investigate linear and nonlinear neural networks for the next-period log-return forecasting. Both models use a certain asset's lagged daily log returns as input features.


We will compare a AR-like simple linear model and a Multilayer Perceptron (MLP) model to investigate if nonlinearity improves model performance. We investigate across several lookback lengths and initialization seeds. The analysis covers Bitcoin (`BTC-USD`), Apple (`AAPL`), and the S&P 500.

The results are mixed. While the linear model performs favourably on Bictoin for the selected test period, it does not generalize consistently to other assets. We also observe that the MLP exhibits larger variations across different random initialization seeds.

This work is exploratory. While we conclude that Linear models could outperform MLP models in certain assets, it in no way suggests that MLP is inherently inferior to a Linear model. Instead, one should interpret the finding of this work as revealing the inadequacy of predicting log return using solely the previous log returns of the corresponding asset---potentially more information is needed to train MLP models.

Potential extension of this project is as follows:
1. Forward validation of the Linear models
2. Adding more input features, including the OHLC prices, not only the close price.
3. Investigate the performance other regression methods, such as Ridge regression with hyperparameters. 
4. Other more complex trading strategy: We use only the sign: $\text{sign}(\hat{r})$. More information could potentially be drawn from the data. 
--- 

## Installation  

Clone the repository:
```bash
git clone https://github.com/sdf1267/NeuralNetworkQuant.git
cd NeuralNetworkQuant
```

Then create a virtual environment:
```bash
python3 -m venv .venv
```

Activate it:

**macOS/Linux**

```bash
source .venv/bin/activate
```

**Windows PowerShell**

```powershell
.venv\Scripts\Activate.ps1
```

Install the project with:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[notebook]"
```
and verify with:

```bash
python -c "import numpy, pandas, matplotlib, torch, yfinance, scipy; print('Installation successful')"
```

### Quick installation for macOS:

```bash
git clone https://github.com/sdf1267/NeuralNetworkQuant.git
cd NeuralNetworkQuant
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[notebook]"
python -m jupyter lab
```


> ## Reproducing figures 
> Open `reproduce_graphs.ipynb` and click `Run All`. The default option will reproduce the results for `BTC-USD`. The simulation may take several minutes to run.

## Research Questions:
1. Could Neural network perform better than simple strategies such as mean reversion? 
2. Does linear model (equivalent to autoregressive time series in the way I implemented) inferior to nonlinear Neural networks such as Multilayer-Perception (MLP) models? 
3. Given the same model, how much would different trading strategies changes the return? 
4. Is the number of features (number of lag days) changing the performance of the model significantly? 

## Method

We compare two Neural networks. Let $X_{t}$ be the asset price at time $t$, and $r_{t} = \ln \left( \frac{X_{t}}{X_{t-1}} \right)$ be the trade log return at time $t$. 

### `LinearModel`
The first model is defined by the class `LinearModel` in the file `networks/LinearModel.py`. It is a single-layer linear Neural network (NN) that maps:

$$
Lin: \underbrace{ \left(r_{t-1}, r_{t-2}\dots  \right)  }_{ n \text{ elements} } \rightarrow r_{t}\, , 
$$

We define $n$ as the _lookback length_: the number of lagged returns provided to our models. This is equivalent to an autoregressive model $AR(n)$ on the log return $r_{t}$ without the noise term: 
$$
\hat{r}_{t} = \sum_{i=1}^{n} \phi_{i}r_{t-i} + c\, , 
$$
where $c$ is the constant bias. 

### `MLPModel`
The second model is again a NN, but instead of a single layer, we use a Multilayer Perceptron (MLP) model with architecture as below

![MLP architecture](networks/fig/mlp_architecture.png)



<!-- ### Data analysis
In training the model, we will split the data for training and testing. We divide the data chronologically into 75% for training and 25% for testing. Since the seeds of  -->

### Strategy
We employ a very simple strategy. Our model takes a set of $n$ lagged data and forecasts the next-period log return.

 - **sign strategy**:  we use the signal $s_{t}$:
$$
s_{t} = \text{sign}(\hat{r}_{t})
$$
where $\hat{r}_{t}$ is the log return predicted by the models. In other words, we only take the direction of  $\hat{r}$ and discard the magnitude from the forecasting.
- **threshold strategy**: Another possibility for the signal is to set: 
	 $$
	s_{t} = \begin{cases}
	+1  &  \hat{r}_{t} > \epsilon  & \text{long}\\
	-1  &  \hat{r}_{t} < -\epsilon   & \text{short}\\
	0  &  \text{otherwise} & \text{hold}
	\end{cases}	
	$$
	here $\epsilon$ is the *threshold*. 

## Simulation protocol

- **Assets:** `BTC-USD`, `AAPL`, and the S&P 500.
- **Frequency:** daily.
- **Sample period** `2019-01-01` to `2026-01-01`.
- **Target:** next-period close-to-close log returns.
- **Features:** the previous $n$-period close-to-close log returns.
- **Split:** take the first 75% for training and the remaining 25% for testing.
- **Models:** A Linear neural network and a nonlinear Multilayer Perceptron model.
- **Transaction Costs:** 10 basis points per position turnover 
- **Strategy:** We use the `threshold strategy` for the data shown below, with `threshold = 0.0001`

<!-- We also have a benchmark strategy, which is a simplified version of mean-reversion strategy:
	3. *mean reversion*: A simple strategy for benchmarking:
		$$
		s_{t} = -\text{sign}(r_{t-1})
		$$
		that is, we bet that the price will move opposite to the previous date. -->

> [!IMPORTANT]
> This is an exploratory work of a complex problem that I will not dare to say that I have solved. Instead, this work merely suggests that some profitability **could be** generated from linear NNs, despite the strong dependencies on the initialization seeds.

## Results
We conclude that the history of individual stock is insufficient for accurate log-return forecasting. The linear model performs favourably on Bitcoin during the selected test period, but similar performance is not observed across other assets. The results should therefore **NOT** be interpreted as evidence of a generally profitable strategy.

Specifically, we have discovered that models have strong dependence on the seed of the random number generators:

### Seed dependence 
We forecast the close-to-close log return of `AAPL`  with transaction cost of 10 basis points and with the __sign strategy__:
1. `seed = 15`
		![alt text](image-7.png)
2. `seed = 55`
        ![alt text](image-8.png)
		

These observations lead us to the folder `networks/average_over_seeds` where we average over randomly chosen seeds to obtain the performance. That means, for each lookback and model, we train and evaluate each model across different initialization seeds.
### Seeds averaging
For each model and lookbacks, we evaluate 50 initialization seeds, randomly chosen from $[0, 10^{6}]$, to obtain the averaged performance of the model. We have transaction cost of 10 basis points. We use the close price of the assets from `2019-01-01` to `2026-01-01`, and train on $75\%$ of the data and test on the rest. 

Let _lookbacks_ be the number of days the NN has access to prior to the forecast date. For example, a lookback length of $5$ means the model receive the previous 5-day close-log-return as input features. We found that the model depends strongly on the lookbacks and the stock of choice. 


### The case of Bitcoin:
![alt text](image-9.png)

We found that MLP has much larger standard deviation across seeds averaging, this may suggest that MLP model needs more data input to reliably obtain model parameters. 

Interestingly, for Bitcoin `BTC-USD`, linear model outperforms MLP. We also have:
![BTC Benchmark](networks/average_over_seeds/results/BTC/benchmark_BTC.png)
we see that in most benchmarks, linear model beats the MLP model. In particular, we have 

1. Pearson and Spearman IC have a small but consistent positive edge.
2. The linear model has a consistent positive annualized Sharpe ratio (note $\sigma_{annualized} = \sqrt{365}\sigma$ since we have Bitcoin). 
3. While Linear model has a lower win rate, which is the number of times when $\text{sign}({\hat{r}_t}) = \text{sign}({r_t})$, the average winning return is larger than the average losing return, leading to a positive total return.

This benchmark suggests that a linear model could produce edge of profitability that survives the assumed transaction costs (10 basis points) within the test period. However, this should be interpreted as elementary results that require further investigation rather than signs of consistent profitability.

__However, I have to emphasize that the model does not consistently generate revenue in all assets.__

### The case of Apple stock `AAPL`
For `AAPL`, neither model result in consistently reliable trading potentials.
![alt text](image-10.png)
The benchmarks are:
![](networks/average_over_seeds/results/AAPL/benchmark_BTC.png)

Here the Linear model has a significantly worse win rate against the MLP model. Similar behaviours can be seen in the S&P 500: 
### The case of the S&P 500 
![alt text](image-11.png)

![alt text](networks/average_over_seeds/results/SNP/benchmark_SNP.png)

Likewise, the S&P 500 does not produce a statistically favoured edge like the Bitcoin. This indicates that the performance of the linear model on Bitcoin is likely not reproducible in other assets.

## Conclusions

To conclude, while an MLP model contains more parameters and nonlinearity than a Linear model, the additional degrees of freedom do not translate to a better performance under the current protocol.

The linear model produces a small positive edge and information coefficient for Bictoin during the selected test period. However, these findings do not generalize to other assets such as `AAPL` or the S&P 500.

Overall, our results that an asset's own lagged log-return only contains weak predictive information. Additional information is likely required to enhance model performances.
