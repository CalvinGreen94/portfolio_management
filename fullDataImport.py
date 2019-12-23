from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split,TimeSeriesSplit
import matplotlib.pyplot as plt
import pandas as pd
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import AdaBoostRegressor
import plotly.plotly as py
import plotly.graph_objs as go
import plotly
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
# %matplotlib inline
import ssl
import json
import ast
import os
import bitfinex
api_v2 = bitfinex.bitfinex_v2.api_v2()
result = api_v2.candles()
import datetime
import time
from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf

data = yf.download("AR", start="2010-01-03", end="2019-12-23")
data = pd.DataFrame(data)
# data = data.drop(['Adj Close'],axis=1)
print(data.tail())
data = data.to_csv('data/stocks/chase_stocks_portfolio/AR.csv')
print('RETRIEVING DAILY STOCK DATA FOR {}'.format(str('AR')))

data = yf.download("CHK", start="2010-01-03", end="2019-12-23")
data = pd.DataFrame(data)
# data = data.drop(['Adj Close'],axis=1)
print(data.tail())
data = data.to_csv('data/stocks/chase_stocks_portfolio/CHK.csv')
print('RETRIEVING DAILY STOCK DATA FOR {}'.format(str('CHK"')))

data = yf.download("PCG", start="2010-01-03", end="2019-12-23")
data = pd.DataFrame(data)
# data = data.drop(['Adj Close'],axis=1)
print(data.tail())
data = data.to_csv('data/stocks/chase_stocks_portfolio/PCG.csv')
print('RETRIEVING DAILY STOCK DATA FOR {}'.format(str('PCG"')))

data = yf.download("SPY", start="2010-01-03", end="2019-12-23")
data = pd.DataFrame(data)
# data = data.drop(['Adj Close'],axis=1)
print(data.tail())
data = data.to_csv('data/stocks/chase_stocks_portfolio/SPY.csv')
print('RETRIEVING DAILY STOCK DATA FOR {}'.format(str('SPY')))

data = yf.download("AAPL", start="2010-01-03", end="2019-12-23")
data = pd.DataFrame(data)
# data = data.drop(['Adj Close'],axis=1)
print(data.tail())
data = data.to_csv('data/stocks/chase_stocks_portfolio/AAPL.csv')
print('RETRIEVING DAILY STOCK DATA FOR {}'.format(str('AAPL')))

data = yf.download("EA", start="2010-01-03", end="2019-12-23")
data = pd.DataFrame(data)
# data = data.drop(['Adj Close'],axis=1)
print(data.tail())
data = data.to_csv('data/stocks/chase_stocks_portfolio/EA.csv')
print('RETRIEVING DAILY STOCK DATA FOR {}'.format(str('EA')))

data = yf.download("FB", start="2010-01-03", end="2019-12-23")
data = pd.DataFrame(data)
# data = data.drop(['Adj Close'],axis=1)
print(data.tail())
data = data.to_csv('data/stocks/chase_stocks_portfolio/FB.csv')
print('RETRIEVING DAILY STOCK DATA FOR {}'.format(str('FB')))

data = yf.download("ROKU", start="2010-01-03", end="2019-12-23")
data = pd.DataFrame(data)
# data = data.drop(['Adj Close'],axis=1)
print(data.tail())
data = data.to_csv('data/stocks/chase_stocks_portfolio/ROKU.csv')
print('RETRIEVING DAILY STOCK DATA FOR {}'.format(str('ROKU')))

data = yf.download("SIRI", start="2010-01-03", end="2019-12-23")
data = pd.DataFrame(data)
# data = data.drop(['Adj Close'],axis=1)
print(data.tail())
data = data.to_csv('data/stocks/chase_stocks_portfolio/SIRI.csv')
print('RETRIEVING DAILY STOCK DATA FOR {}'.format(str('SIRI')))

data = yf.download("XOM", start="2010-01-03", end="2019-12-23")
data = pd.DataFrame(data)
# data = data.drop(['Adj Close'],axis=1)
print(data.tail())
data = data.to_csv('data/stocks/chase_stocks_portfolio/XOM.csv')
print('RETRIEVING DAILY STOCK DATA FOR {}'.format(str('XOM')))

data = yf.download("^GSPC", start="2010-01-03", end="2019-12-23")
data = pd.DataFrame(data)
# data = data.drop(['Adj Close'],axis=1)
print(data.tail())
data = data.to_csv('data/stocks/chase_stocks_portfolio/^GSPC.csv')
print('RETRIEVING DAILY STOCK DATA FOR {}'.format(str('^GSPC')))
# plotly.tools.set_credentials_file(username='Gamma-AI1011', api_key='KoXH9I7ffpwUueVaa7TT')
# Cov = pd.read_csv("futures_data/6B 09-19.Last.txt", sep=';', header=None)
# Cov.columns = ["yyyyMMdd", "open price", "high price", "low price",'close price','volume Pl'] # "yyyyMMdd" = MINUTE BY MINUTE
# data = Cov.to_csv('data/futures/6B09-19.Last.csv')
# # data = pd.read_csv('data/futures/6B09-19.Last.csv')
# # data = data.drop(['Unnamed: 0'], axis=1)
# # short_rolling = data.rolling(window=20).mean()
# # long_rolling = data.rolling(window=100).mean()
# # start_date= '37265 20190812 045000'
# # end_date = '51423  20190904 182300'
# # long_rolling.tail()
# # short_rolling.head(20)
# # fig, ax = plt.subplots(figsize=(16,9))
# # # my_year_month_fmt = mdates.DateFormatter('%m/%y')
# # ax.plot(data.loc[start_date:end_date, :].index, data.loc[start_date:end_date, 'low price'], label='Price')
# # ax.plot(long_rolling.loc[start_date:end_date, :].index, long_rolling.loc[start_date:end_date, 'low price'], label = '100-days SMA')
# # ax.plot(short_rolling.loc[start_date:end_date, :].index, short_rolling.loc[start_date:end_date, 'low price'], label = '20-days SMA')
# # ax.legend(loc='best')
# # ax.set_ylabel('Price in $')
# # # ax.xaxis.set_major_formatter(my_year_month_fmt)
# # plt.show()
# # plt.close()
# # df0,df1 = data.shape[0], data.shape[1]
# # data = data.drop(['Unnamed: 0'],axis=1)
# # print('DATA HAS {} TRANSACTIONS BY THE MINUTE WITH {} FEATURES'.format(df0,df1))
# # data = data.drop(['yyyyMMdd'],axis=1)
# # high = data['high price']
# # low = data['low price']
# # data.describe()
# # X= data.drop(['open price'],axis=1)
# # y= data['open price']
# # mini = MinMaxScaler()
# # X = mini.fit_transform(X)
# # X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.67,shuffle=False)
# # reg = LinearRegression(normalize=True,n_jobs=-1)
# # fit = reg.fit(X_train,y_train)
# # score = reg.score(X_test,y_test)
# # print('score for 6B {}'.format(score))
# # pred = reg.predict(X_test[[-1]])
# # pred = pd.DataFrame(pred)
# # prev = y_test
# # prev = pd.DataFrame(prev)
# # print('6B PREVIOUS OPEN'.format(prev))
# # print('6B PREDICTED OPEN {}\n'.format(pred))
#
# Cov = pd.read_csv("futures_data/6E 09-19.Last.txt", sep=';', header=None)
# Cov.columns = ["yyyyMMdd", "open price", "high price", "low price",'close price','volume Pl'] # "yyyyMMdd" = MINUTE BY MINUTE
# data = Cov.to_csv('data/futures/6E09-19.Last.csv')
# # data = pd.read_csv('data/futures/6E09-19.Last.csv')
# # data = data.drop(['Unnamed: 0'], axis=1)
# # short_rolling = data.rolling(window=20).mean()
# # long_rolling = data.rolling(window=100).mean()
# # start_date= '7059 20190819 031000'
# # end_date = '7359 20190819 083300'
# # long_rolling.tail()
# # short_rolling.head(20)
# # fig, ax = plt.subplots(figsize=(16,9))
# # # my_year_month_fmt = mdates.DateFormatter('%m/%y')
# # ax.plot(data.loc[start_date:end_date, :].index, data.loc[start_date:end_date, 'low price'], label='Price')
# # ax.plot(long_rolling.loc[start_date:end_date, :].index, long_rolling.loc[start_date:end_date, 'low price'], label = '100-days SMA')
# # ax.plot(short_rolling.loc[start_date:end_date, :].index, short_rolling.loc[start_date:end_date, 'low price'], label = '20-days SMA')
# # # ax.legend(loc='best')
# # ax.set_ylabel('Price in $')
# # # ax.xaxis.set_major_formatter(my_year_month_fmt)
# # plt.show()
# # plt.close()
# # data = data.drop(['Unnamed: 0'],axis=1)
# # df0,df1 = data.shape[0], data.shape[1]
# # print('DATA HAS {} TRANSACTIONS BY THE MINUTE WITH {} FEATURES'.format(df0,df1))
# # data1 = data.drop(['yyyyMMdd'],axis=1)
# # high1= data1['high price']
# # low1 = data1['low price']
# # data1.describe()
# # def train_test1():
# #     X= data1.drop(['open price'],axis=1)
# #     y= data1['open price']
# #     returns = data1.pct_change()[1:]
# #     mini = MinMaxScaler()
# #     X = mini.fit_transform(X)
# #     X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.67,shuffle=False)
# #     reg = LinearRegression(normalize=True,n_jobs=-1)
# #     fit = reg.fit(X_train,y_train)
# #     returns = returns[[-long_ma]]
# #     print(returns)
# #     spreads = returns - (reg.coeff() * returns)
# #     print(spreads)
# #     score = reg.score(X_test,y_test)
# #     pred = reg.predict(X_test[[-1]])
# #     print('6E score {}'.format(score))
# #     print('6E PREDICTED OPEN {} \n'.format(pred))
# #     return X_train,X_test,y_train,y_test,fit,pred,score,returns, spreads
# # train_test1()
#
# Cov = pd.read_csv("futures_data/CL 09-19.Last.txt", sep=';', header=None)
# Cov.columns = ["yyyyMMdd", "open price", "high price", "low price",'close price','volume Pl'] #EOD TRADING IS USED , "yyyyMMdd" = MINUTE BY MINUTE
# data = Cov.to_csv('data/futures/CL09-19.Last.csv')
# # data = pd.read_csv('CL 09-19.Last.txt')
# # data = data.drop(['Unnamed: 0'],axis=1)
# # df0,df1 = data.shape[0], data.shape[1]
# # print('DATA HAS {} TRANSACTIONS BY THE MINUTE WITH {} FEATURES'.format(df0,df1))
# # data2 = data.drop(['yyyyMMdd'],axis=1)
# # high2 =data2['high price']
# # low2 = data2['low price']
# # data2.describe()
# # def train_test2():
# #     X= data2.drop(['open price'],axis=1)
# #     y= data2['open price']
# #     mini = MinMaxScaler()
# #     X = mini.fit_transform(X)
# #     X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.67,shuffle=False)
# #     reg = LinearRegression(normalize=True,n_jobs=-1)
# #     fit = reg.fit(X_train,y_train)
# #     score = reg.score(X_test,y_test)
# #     pred = reg.predict(X_test[[-1]])
# #     print('CL 9-19 score {}'.format(score))
# #     print('CL 9-19 PREDICTED OPEN {} \n'.format(pred))
# #     return X_train,X_test,y_train,y_test,fit,pred,score
# # train_test2()
#
# Cov = pd.read_csv("futures_data/CL 10-19.Last.txt", sep=';', header=None)
# Cov.columns = ["yyyyMMdd", "open price", "high price", "low price",'close price','volume Pl'] #EOD TRADING IS USED , "yyyyMMdd" = MINUTE BY MINUTE
# data = Cov.to_csv('data/futures/CL10-19.Last.csv')
# # data = pd.read_csv('CL 10-19.Last.txt')
# # data = data.drop(['Unnamed: 0'],axis=1)
# # df0,df1 = data.shape[0], data.shape[1]
# # print('DATA HAS {} TRANSACTIONS BY THE MINUTE WITH {} FEATURES'.format(df0,df1))
# # data3 = data.drop(['yyyyMMdd'],axis=1)
# # high3= data3['high price']
# # low3= data3['low price']
# # data3.describe()
# # def train_test3():
# #     X= data3.drop(['open price'],axis=1)
# #     y= data3['open price']
# #     mini = MinMaxScaler()
# #     X = mini.fit_transform(X)
# #     X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.67,shuffle=False)
# #     reg = LinearRegression(normalize=True,n_jobs=-1)
# #     fit = reg.fit(X_train,y_train)
# #     score = reg.score(X_test,y_test)
# #     pred = reg.predict(X_test[[-1]])
# #     print('CL 10-19 score {}'.format(score))
# #     print('predicted CL 10-19 PREDICTED OPEN {} \n'.format(pred))
# #     return X_train,X_test,y_train,y_test,fit,pred,score
# # train_test3()
#
# Cov = pd.read_csv("futures_data/GC 12-19.Last.txt", sep=';', header=None)
# Cov.columns = ["yyyyMMdd", "open price", "high price", "low price",'close price','volume Pl'] #EOD TRADING IS USED , "yyyyMMdd" = MINUTE BY MINUTE
# data = Cov.to_csv('data/futures/GC12-19.Last.csv')
# # data = pd.read_csv('GC 12-19.Last.txt')
# # data.head()
# # data = data.drop(['Unnamed: 0'],axis=1)
# # df0,df1 = data.shape[0], data.shape[1]
# # print('DATA HAS {} TRANSACTIONS BY THE MINUTE WITH {} FEATURES'.format(df0,df1))
# # data4 = data.drop(['yyyyMMdd'],axis=1)
# # high4= data4['high price']
# # low4 = data4['low price']
# # data4.describe()
# # def train_test4():
# #     X= data4.drop(['open price'],axis=1)
# #     y= data4['open price']
# #     mini = MinMaxScaler()
# #     X = mini.fit_transform(X)
# #     X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.67,shuffle=False)
# #     reg = LinearRegression(normalize=False,n_jobs=-1)
# #     fit = reg.fit(X_train,y_train)
# #     pred = reg.predict(X_test[[-1]])
# #     score = reg.score(X_test,y_test)
# #     print('GC 12-19 score {}'.format(score))
# #     print('GC 12-19 PREDICTED OPEN {} \n'.format(pred))
# #     return X_train,X_test,y_train,y_test,fit,pred,score
# # train_test4()
# #
# Cov = pd.read_csv("futures_data/NQ 09-19.Last.txt", sep=';', header=None)
# Cov.columns = ["yyyyMMdd", "open price", "high price", "low price",'close price','volume Pl'] #EOD TRADING IS USED , "yyyyMMdd" = MINUTE BY MINUTE
# data = Cov.to_csv('data/futures/NQ09-19.Last.csv')
# # data = pd.read_csv('NQ 09-19.Last.txt')
# # data = data.drop(['Unnamed: 0'],axis=1)
# # data = data.drop(['yyyyMMdd'],axis=1)
# # df0,df1 = data.shape[0], data.shape[1]
# # print('NQ futures Data Has {} Transactions with {} features'.format(df0,df1))
# # data.describe()
# # X= data.drop(['open price'],axis=1)
# # y= data['open price']
# # mini = MinMaxScaler()
# # X = mini.fit_transform(X)
# # X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.67,shuffle=False)
# # reg = LinearRegression(normalize=True,n_jobs=-1)
# # reg.fit(X_train,y_train)
# # print('NQ score {}'.format(reg.score(X_test,y_test)))
# # print('NQ PREDICTED OPEN  {} \n'.format(reg.predict(X_test[[-1]])))
#
# Cov = pd.read_csv("futures_data/ZB 09-19.Last.txt", sep=';', header=None)
# Cov.columns = ["yyyyMMdd", "open price", "high price", "low price",'close price','volume Pl'] #EOD TRADING IS USED , "yyyyMMdd" = MINUTE BY MINUTE
# data = Cov.to_csv('data/futures/ZB09-19.Last.csv')
# # data = pd.read_csv('ZB 09-19.Last.txt')
# # data = data.drop(['Unnamed: 0'],axis=1)
# # df0,df1 = data.shape[0], data.shape[1]
# # print('DATA HAS {} TRANSACTIONS BY THE MINUTE WITH {} FEATURES'.format(df0,df1))
# # data5 = data.drop(['yyyyMMdd'],axis=1)
# # high5 =  data5['high price']
# # low5 = data5['low price']
# # data5.describe()
# # def train_test5():
# #     X= data5.drop(['open price'],axis=1)
# #     y= data5['open price']
# #     mini = MinMaxScaler()
# #     X = mini.fit_transform(X)
# #     X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.67,shuffle=False)
# #     reg = LinearRegression(normalize=True,n_jobs=-1)
# #     fit = reg.fit(X_train,y_train)
# #     pred = reg.predict(X_test[[-1]])
# #     score = reg.score(X_test,y_test)
# #     print('ZB score {}'.format(score))
# #     print('ZB PREDICTED OPEN {} \n '.format(pred))
# #     return X_train,X_test,y_train,y_test,fit,pred,score
# # train_test5()
#
print('DOWNLOADING BITCOIN PAIRS DATA')
api_v2 = bitfinex.bitfinex_v2.api_v2()
result = api_v2.candles()
time_step = 60000000
 # Define query parameters
pair = 'btcusd' # Currency pair of interest
bin_size = '1d' # This will return minute data
limit = 1000    # We want the maximum of 1000 data points
# Define the start date
t_start = datetime.datetime(2010, 1, 3, 0, 0) 
t_start = time.mktime(t_start.timetuple()) * 1000
# Define the end date
t_stop = datetime.datetime(2019, 12, 23, 0, 0)
t_stop = time.mktime(t_stop.timetuple()) * 1000
result = api_v2.candles(symbol=pair, interval=bin_size,
                        limit=limit, start=t_start, end=t_stop)
# result =  pd.DataFrame(list_of_rows,columns=['PRICES','PRICE:'])
def fetch_data(start, stop, symbol, interval, tick_limit, step):
    api_v2 = bitfinex.bitfinex_v2.api_v2()
    data = []
    start = start - step
    while start < stop:
        start = start + step
        end = start + step
        res = api_v2.candles(symbol=symbol, interval=interval,
                             limit=tick_limit, start=start,
                             end=end)
        data.extend(res)
        time.sleep(2)
    return data
api_v1 = bitfinex.bitfinex_v1.api_v1()
pairs = ['btcusd','xtzusd','oxtusd','xlmusd','xrpusd','zecusd','ethusd','etcusd','ltcusd','daiusd','eosusd']#api_v1.symbols()
save_path = 'data/crypto/chase_crypto_portfolio'
if os.path.exists(save_path) is False:
    os.mkdir(save_path)
for pair in pairs:
    pair_data = fetch_data(start=t_start, stop=t_stop, symbol=pair, interval=bin_size, tick_limit=limit, step=time_step)
    # Remove error messages
    ind = [np.ndim(x) != 0 for x in pair_data]
    pair_data = [i for (i, v) in zip(pair_data, ind) if v]
    #Create pandas data frame and clean data
    names = ['time', 'open', 'high', 'low', 'close', 'volume']
    df = pd.DataFrame(pair_data, columns=names)
    df.drop_duplicates(inplace=True)
    # df['time'] = pd.to_datetime(df['time'], unit='ms')
    df.set_index('time', inplace=True)
    df.sort_index(inplace=True)
    print('Done downloading data. Saving to .csv.')
    df.to_csv('{}/bitfinex_{}.csv'.format(save_path, pair))
    print('Done saving pair{}. Moving to next pair.'.format(pair))
    # df.drop(['volume'],axis=1)
print('Done retrieving data')
