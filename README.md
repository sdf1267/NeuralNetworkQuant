

# Neural Network Pricing
Pricing prediction based on neural networks. 


--- 

## Planned Features

- [x] Linear model NN model for training 
- [x] Predict stock prices with NN 
- [x] Extend to nonlinear models

---

## Research Questions:
- [ ] Could Neural network perform better than simple strategies such as mean reversion? 
- [ ] Does linear model (equivalent to autoregressive time series in the way I implemented) inferior to nonlinear Neural networks such as Multilayer-Perception (MLP) models? 
- [ ] Given the same model, how much would different trading strategies changes the return? 
- [ ] Is the number of features (number of lag days) changing the performance of the model significantly?


---
## Method

We compare two Neural networks. Let $X_{t}$ be the asset price at time $t$, and $r_{t} = \ln \left( \frac{X_{t}}{X_{t-1}} \right)$ be the trade log return at time $t$. 

### `LinearModel`
The first model is defined by the class `LinearModel` in the file `networks/LinearModel.py`. It is a single-layer linear Neural network (NN) that maps:

$$
Lin: \underbrace{ \left(r_{t-1}, r_{t-2}\dots  \right)  }_{ n \text{ elements} } \rightarrow r_{t}\, , 
$$

this is equivalent to a autoregressive model $AR(n)$ on the log return $r_{t}$ without the noise term: 
$$
\hat{r}_{t} = \sum_{i=1}^{n} \phi_{i}r_{t-i} + c\, , 
$$
where $c$ is the constant bias. 

### `MPLModel`
The second model is again a NN, but instead of a single layer, we use a Multilayer Perceptron (MLP) model with architecture as below

![MLP architecture](networks/fig/mlp_architecture.png)
