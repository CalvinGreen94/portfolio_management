import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import pandas as pd
style.use('fivethirtyeight')
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
# prints formatted price
def moving_20average(a, n=25) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def moving_100average(a, n=100) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
# a = data
# print(moving_average(a))
# low_moving = pd.DataFrame(moving_average(a))
# low_moving = low_moving.to_csv('LOW_SIMPLE_MOVING_AVG.csv')
def formatPrice(n):
	return ("-$" if n < 0 else "$") + "{0:.4f}".format(abs(n))
# returns the vector containing stock data from a fixed file

def getStockDataVec(key):
	vec = []
	lines = open("data/" + key + ".csv", "r").read().splitlines()
	print(len(lines))

	for line in lines[1:]:
		vec.append(float(line.split(",")[3])) #HIGH
		# print('initializing 20 second moving average')
		# a = moving_20average(vec)
		# print('initializing 100 second moving average')
		# b = moving_100average(vec)
	# for ma in ma20:
	# 	vec.append(float(line.split(',')[4]))
	# for ma1, in ma100:
	# 	vec.append(float(line.split(',')[4]))

	ax1.clear()
	ax1.plot(vec)
	# ax1.plot(a)
	# ax1.plot(b)
	ax1
#     ani = animation.FuncAnimation(fig, vec, interval=1000)
	plt.show()
	return vec

# returns the sigmoid
def sigmoid(x):
	return 1 / (1 + math.exp(-x))

# returns an an n-day state representation ending at time t
def getState(data, t, n):
	d = t - n + 1
	block = data[d:t + 1] if d >= 0 else -d * [data[0]] + data[0:t + 1] # pad with t0
	res = []
	for i in range(n - 1):
		res.append(sigmoid(block[i + 1] - block[i]))

	return np.array([res])
