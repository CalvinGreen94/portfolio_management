## Overview

This is the ProjectGamma.AI Python 3.6 code for implementing a Deep Q neural networks for algorithmic trading within the Futures market.It's implementation of Q-learning applied to (short-term) stock trading. The model uses n-day windows of LOW prices (OHLC) to determine if the best action to take at a given time is to buy, sell or hold.


## Results

Results of the examples included in datasets feature rewards, test scores, and actions for the state of a given stock price. Also, implematation of forecasting  will be considered for more precise actions into the future of the market.


## Running the Code
** = STOCK/CRYPTO
_ = window(# of days)
-- = episodes

Step 1: python fulldataimport.py

step1 will extract the text files and real time intraday cryptocurrency data into the data format suitable for the machine learning simulation environment for the Deep Q Network.

2:python train.py data/(crypto/** or stocks/) _ -
step2 minimum 200 episodes for results

python evaluate.py  _  model_ep--

## WIP:
test simulation results against benchmark regressor while analyzing data within 15 minute - 9 year windows. LINEAR REGRESSOR IPYNB NOTEBOOK WILL BE THE BENCHMARK MODEL USING LINEAR REGRESSION AND TIME SERIES CROSS VALIDATION.. AND COMPARE IT TO THE BACKTEST SIMULATION RESULTS 
