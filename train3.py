from agent3.agent import Agent
from functions import *
import sys
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import csv
os.environ['KERAS_BACKEND' ] = 'tensorflow'
os.environ['MKL_THREADING_LAYER'] = 'GNU'
from matplotlib import style
import pandas as pd
if len(sys.argv) != 4:
	print("Usage: python train.py [stock] [window] [episodes]")
	exit()
stock_name, window_size, episode_count = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
agent = Agent(window_size)
data = getStockDataVec(stock_name)
l = len(data) - 1
batch_size = 710
ma20 = moving_20average(data)
ma100 = moving_100average(data)
# plt.plot(ma20,c='black')
# plt.plot(ma100,c = 'white')
plt.subplot(2, 1, 1)
plt.plot(ma20,c='black')
plt.plot(ma100,c='white')
plt.ylabel('price')

plt.subplot(2, 1, 2)
plt.plot(data)
plt.xlabel('time (m)')
plt.ylabel('price')
plt.show()
plt.close
for e in range(episode_count + 1):
	print("Episode " + str(e) + "/" + str(episode_count))
	state = getState(data, 0, window_size + 1)
	total_profit = 0
	agent.inventory = []
	starting_balance = 6.27 #10.00000
	print('starting balance {}'.format(starting_balance))
	buying_power = 12
	for t in range(l):
		action = agent.act(state)
		# hold
		next_state = getState(data, t + 1, window_size + 1)
		reward = 1.
		#buy
		if action == 1:
			agent.inventory.append(data[t])
			print('CURRENT PRICE ${:.5f}'.format(data[t]))
			print("Buy: " + formatPrice(data[t]*starting_balance/buying_power))
			print('Current Balance ${:.5f}'.format(data[t]-starting_balance))
		# sell
		elif action == 2 and len(agent.inventory) > 0:
			bought_price = agent.inventory.pop(0)
			reward = max(data[t] - bought_price, 0)
			print('REWARD\n ', reward)
			total_profit += data[t] - bought_price
			print('CURRENT PRICE {}'.format([data[t]]))
			print("Sell: " + formatPrice(data[t]) + " | Profit: " + formatPrice(data[t] - bought_price))
			print("--------------------------------")
			print("Total Profit: " + formatPrice(total_profit))
			print("--------------------------------")
			print('CURRENT BALANCE ${:.5f}'.format(starting_balance+total_profit))

		# a2 = pd.DataFrame(action)
		# a2 = pd.to_csv('SellPrice.csv')
		done = True if t == l - 1 else False
		agent.memory.append((state, action, reward, next_state, done))

		state = next_state
		s = pd.DataFrame(state)
		s = s.to_csv('state.csv')
		if done:
			print("--------------------------------")
			print("Total Profit: " + formatPrice(total_profit))
			print("--------------------------------")

		if len(agent.memory) > batch_size:
			agent.expReplay(batch_size)

	if e % 10 == 0:
		agent.model.save("models/agent3/model_ep-" + str(e))
