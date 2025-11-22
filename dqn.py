import utils
import gymnasium as gym
import numpy as np
import tensorflow as tf
import keras
import random
from collections import deque

utils.init()

class DQNAgent:
    def __init__(self, env, state_size, action_size):
        self.env = env
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen = 2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        model = keras.Sequential()
        model.add(keras.layers.Dense(32, input_dim=self.state_size, activation = 'relu'))
        model.add(keras.layers.Dense(32, activation = 'relu'))
        model.add(keras.layers.Dense(self.action_size, activation='softmax'))
        model.compile(loss = 'mse', optimizer = "adam")

        return model
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        act_values = self.model.predict(state, verbose = 0) # type: ignore
        return np.argmax(act_values[0])
    
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state, verbose = 0)

            if done:
                target[0][action] = reward
            else:
                t = self.target_model.predict(next_state, verbose = 0)
                target[0][action] = reward + self.gamma * np.amax(t[0])

            self.model.fit(state, target, epochs = 1, verbose = 0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())