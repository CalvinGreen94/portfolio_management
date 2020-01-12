## Overview

This is the ProjectGamma.AI Python 3.6 code for implementing a Deep Q neural networks for algorithmic trading within the Futures market.It's implementation of Q-learning applied to (short-term) stock trading. The model uses n-day windows of High prices (Time,Open,High,Low,Close,Volume) to determine if the best action to take at a given time is to buy, sell or hold.


## Results

Results of the examples included in datasets feature rewards, test scores, and actions for the state of a given stock price. 


## Running the Code


75 = window(# of days)


200 = episodes

Step 1: python fulldataimport.py


--> OPEN "Linear_Reg__.ipynb" -- this is the benchmark model for analyzing and predicted time segments between 15-5 minutes.
(if in a rush remove the stock import feed)

step1 will extract the text files and real time intraday cryptocurrency data into the data format suitable for the machine learning simulation environment for the Deep Q Network.

2:

portfolio_management > python train2.py crypto/crypto_portfolio/15m/bitfinex_ethusd 75 200

portfolio_management > python train2.py crypto/crypto_portfolio/5m/bitfinex_ethusd 25 200

minimum 200 episodes for results

portfolio_management > python evaluate2.py  crypto/crypto_portfolio/15m/bitfinex_ethusd  agent2/model_ep-200

portfolio_management > python evaluate2.py  crypto/crypto_portfolio/5m/bitfinex_ethusd  agent2/model_ep-200

## WIP:

Test simulation results against benchmark regressor while analyzing data within 15 minute -5 minute windows. LINEAR REGRESSOR IPYNB NOTEBOOK WILL BE THE BENCHMARK MODEL USING LINEAR REGRESSION AND TIME SERIES CROSS VALIDATION.. AND COMPARE IT TO THE BACKTEST SIMULATION RESULTS. Add in psychology behind the trades. Add indicators (news (sentiment analysis, Boilinger bands, Volume Indicator)).
