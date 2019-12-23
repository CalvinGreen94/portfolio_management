import keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.optimizers import Adam,RMSprop
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from keras.utils import to_categorical
import os
os.environ['KERAS_BACKEND' ] = 'tensorflow'
os.environ['MKL_THREADING_LAYER'] = 'GNU'
from keras.callbacks import TensorBoard
import pandas as pd
import numpy as np
from keras import optimizers
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dropout, Activation
from sklearn.preprocessing import MinMaxScaler
import time
import numpy as np
import random
from collections import deque
class Agent:
	def __init__(self, state_size, is_eval=False, model_name=""):
		self.state_size = state_size # normalized previous days
		self.action_size = 3 # sit, buy, sell
		self.memory = deque(maxlen=1020)
		self.inventory = []
		self.model_name = model_name
		self.is_eval = is_eval
		self.gamma = 0.88
		self.epsilon = 1.0
		self.epsilon_min = 0.25
		self.epsilon_decay = 0.9999
		self.model = load_model("models/" + model_name) if is_eval else self._model()

	def _model(self):
		batch_size = 666
		dr = 0.75

		visible = Input(shape=(self.state_size,))
		hidden1 = Dense(5, activation='sigmoid')(visible)
		hidden1 = Dense(5, activation='sigmoid')(hidden1)
		hidden1 = Dense(5, activation='sigmoid')(hidden1)
		hidden1 = Dropout(dr)(hidden1)

		hidden2 = Dense(5, activation='sigmoid')(hidden1)
		hidden2 = Dense(5, activation='sigmoid')(hidden2)
		hidden2 = Dense(5, activation='sigmoid')(hidden2)
		hidden2 = Dropout(dr)(hidden2)

		hidden3 = Dense(5, activation='sigmoid')(hidden2)
		hidden3 = Dense(5, activation='sigmoid')(hidden3)
		hidden3 = Dropout(dr)(hidden3)

		merge = keras.layers.concatenate([hidden1,hidden2,hidden3], axis=1)
		hidden4 = Dense(self.action_size, activation="relu")(merge)
		output =Dense(self.action_size, activation="sigmoid")(hidden4)
		model = Model(inputs=visible, outputs=output)
		model.compile(loss='mse',optimizer='rmsprop')
		return model

	def act(self, state):
		if not self.is_eval and random.random() <= self.epsilon:
			return random.randrange(self.action_size)
		action = self.model.predict(state)
		return np.argmax(action[0])

	def expReplay(self, batch_size):
		mini_batch = []
		l = len(self.memory)
		for i in range(l - batch_size + 1, l):
			mini_batch.append(self.memory[i])
		for state, action, reward, next_state, done in mini_batch:
			target = reward
			if not done:
				target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])

			target_f = self.model.predict(state)
			target_f[0][action] = target
			self.model.fit(state, target_f, epochs=1, verbose=0,validation_split=.65)

		if self.epsilon > self.epsilon_min:
			self.epsilon *= self.epsilon_decay
