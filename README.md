## Overview

This is the ProjectGamma.AI Python 3.6 code for implementing a Deep Q neural networks for algorithmic trading within the Futures market.It's implementation of Q-learning applied to (short-term) stock trading. The model uses n-day windows of LOW prices (OHLCV) to determine if the best action to take at a given time is to buy, sell or hold.

As a result of the short-term state representation, the model is not very good at making decisions over long-term t. Implementing multiple DQN agents over various market tickers.. The main thing to consider is the size of the data and how it will be engineered to achieve the most optimal predicted future price.

## Results

Results of the examples included in datasets feature rewards, test scores, and actions for the state of a given stock price. Also, implematation of forecasting  will be considered for more precise actions into the future of the market. This model will be used for the AI trading bot via coinbase api keys.


## Running the Code
** = STOCK/CRYPTO
_ = window(# of days)
-- = episodes
directory :

--- cd: path/to/portfolio_management
-- cd portfolio_management
- portfolio_management> python fulldataimport.py

step1 will extract the text files and real time intraday cryptocurrency data into the data format suitable for the machine learning simulation environment for the Deep Q Network.

portfolio_management> python train#.py crypto/** oror stocks/) 200 (windows of days) 200 (episodes)
step2 minimum 200 episodes for results

python evaluate.py  200  model_ep200

## WIP:
test simulation results against benchmark regressor while analyzing data within 15 minute - 9 year windows. LINEAR REGRESSOR IPYNB NOTEBOOK WILL BE THE BENCHMARK MODEL USING LINEAR REGRESSION AND TIME SERIES CROSS VALIDATION.. AND COMPARE IT TO THE BACKTEST SIMULATION RESULTS. 
