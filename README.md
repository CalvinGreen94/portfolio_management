## Overview

This is Python 3.6 code for implementing a Deep Q neural networks for algorithmic trading within the cryptocurrency/stock market.It's implementation of Q-learning applied to (short-term) trading. The model uses n-day windows of High prices (Given that is the quantum algorithm decision for choosing the feature at the time) (Time,Open,High,Low,Close,Volume) to determine if the best action to take at a given time is to buy, sell or hold.


## Results

Results of the examples included in datasets feature rewards, test scores, and actions for the state of a given stock price. 


## Running the Code


75 = window(# of days)


200 = episodes

Step 1:

portfolio_management>python fulldataimport.py

(if in a rush remove the stock import feed)

step1 will extract the text files and real time intraday cryptocurrency data into the data format suitable for the machine learning simulation environment for the Deep Q Network.

2:


--> OPEN "01-feature-selection.ipynb". This is the quantum feature selection algorithm


--> OPEN "Linear_Reg__.ipynb" -- this is the benchmark model for analyzing and predicted time segments between 15-5 minutes.


portfolio_management > python train2.py crypto/crypto_portfolio/15m/bitfinex_ethusd 75 200

portfolio_management > python train2.py crypto/crypto_portfolio/5m/bitfinex_ethusd 25 200

minimum 200 episodes for results

TRAINING MODELS CAN BE RAN AT THE SAME TIME BUT HIGHLY UNRECOMMENDED. 


portfolio_management > python evaluate2.py  crypto/crypto_portfolio/15m/bitfinex_ethusd  agent2/model_ep-200

portfolio_management > python evaluate2.py  crypto/crypto_portfolio/5m/bitfinex_ethusd  agent2/model_ep-200

## WIP:

Test simulation results against benchmark regressor while analyzing data within 15 minute -5 minute windows. LINEAR REGRESSOR IPYNB NOTEBOOK WILL BE THE BENCHMARK MODEL USING LINEAR REGRESSION AND TIME SERIES CROSS VALIDATION.. AND COMPARE IT TO THE BACKTEST SIMULATION RESULTS. Add in psychology behind the trades. Add indicators (news (sentiment analysis, Boilinger bands, Volume Indicator)).
