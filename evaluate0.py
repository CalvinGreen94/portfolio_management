import keras
from keras.models import load_model
import os
from coinbase.wallet.client import Client
import json
# Before implementation, set environmental variables with the names API_KEY and API_SECRET
api_key = 'u6Ni9BqAj46es5jE'
api_secret = 'BDo0b1p3UsiTCqhHjYMprsfSE9mS9IxL'

client = Client(api_key, api_secret)
user = client.get_current_user()
# user_as_json_string = json.dumps(user)
# accounts = client.get_accounts()
# assert isinstance(accounts.data, list)
# assert accounts[0] is accounts.data[0]
# assert len(accounts[::]) == len(accounts.data)
accounts = client.get_account('a5169272-8826-58ca-9f7d-b827785f7cd6')
# accounts = client.get_accounts()
print(accounts)
assert (accounts.warnings is None) or isinstance(accounts.warnings, list)
# accounts = client.get_accounts()
assert (accounts.pagination is None) or isinstance(accounts.pagination, dict)
print(client.get_buy_price(currency_pair = 'XRP-USD'))
print(client.get_sell_price(currency_pair = 'XRP-USD'))
# client.deposit(account_id, amount='10', currency='USD')
# client.buy(account_id, amount='5', currency='XRP')
# client.sell(account_id, amount='1', currency='XRP')
os.environ['KERAS_BACKEND' ] = 'tensorflow'
os.environ['MKL_THREADING_LAYER'] = 'GNU'
from agent0.agent import Agent
from functions import *
import sys

if len(sys.argv) != 3:
	print("Usage: python evaluate.py [stock] [model]")
	exit()

stock_name, model_name = sys.argv[1], sys.argv[2]
model = load_model("models/agent0" + model_name)
window_size = model.layers[0].input.shape.as_list()[1]

agent = Agent(window_size, True, model_name)
data = getStockDataVec(stock_name)
l = len(data) - 1
batch_size = 32

state = getState(data, 0, window_size + 1)
total_profit = 0
agent.inventory = []
starting_balance = .2961 #10.00000
print('starting balance {}'.format(starting_balance))
buying_power = .0002
for t in range(l):
	action = agent.act(state)
	# sit
	next_state = getState(data, t + 1, window_size + 1)
	reward = 0

	if action == 1: # buy
		agent.inventory.append(data[t])
		print("Buy: " + formatPrice(data[t]))

	elif action == 2 and len(agent.inventory) > 0: # sell
		bought_price = agent.inventory.pop(0)
		reward = max(data[t] - bought_price, 0)
		total_profit += data[t] - bought_price
		print("Sell: " + formatPrice(data[t]) + " | Profit: " + formatPrice(data[t] - bought_price))
		print("--------------------------------")
		print("Total Profit: " + formatPrice(total_profit))
		print("--------------------------------")
	done = True if t == l - 1 else False
	agent.memory.append((state, action, reward, next_state, done))
	state = next_state
	if done:
		print("--------------------------------")
		print(stock_name + " Total Profit: " + formatPrice(total_profit))
		print("--------------------------------")
