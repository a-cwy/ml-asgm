import gymnasium as gym
import numpy as np
import keras
import random

from collections import deque

class DQNAgent():
    def __init__(
            self, 
            env: gym.Env, 
            input_size, 
            output_size, 
            learning_rate = 0.001, 
            epsilon = 1,
            min_epsilon = 0.01,
            epsilon_decay = 0.995,
            discount_factor = 0.9,
            mini_batch_size = 128,
            replay_memory_size = 1500 
        ):

        self.env = env
        self.input_size = input_size
        self.output_size = output_size

        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.discount_factor = discount_factor
        self.mini_batch_size = mini_batch_size
        
        self.memory = deque([], maxlen = replay_memory_size)

        self.policy_network = self._build_model(input_size, output_size)
        self.target_network = self._build_model(input_size, output_size)
        self._sync_network()

    

    def load_models(self, policy_path, target_path):
        self.policy_network = keras.models.load_model(policy_path)
        self.target_network = keras.models.load_model(target_path)



    def _build_model(self, input_size, output_size):
        model = keras.Sequential()
        model.add(keras.layers.Input((input_size,)))
        model.add(keras.layers.Dense(16, activation = "relu"))
        model.add(keras.layers.Dense(output_size, activation = "linear"))
        model.compile(loss = 'mse', optimizer = keras.optimizers.Adam(learning_rate = self.learning_rate))

        return model
    


    def _sync_network(self):
        self.target_network.set_weights(self.policy_network.weights)



    def _flatten_obs(self, obs):
        return np.array(list(obs.values())).reshape(1, self.input_size)



    def _optimize(self, mini_batch):
        for obs, reward, action, next_obs, done in mini_batch:
            tarnet_out = self.target_network.predict(self._flatten_obs(obs), verbose = 0)
            q_new = reward if done else reward + self.discount_factor * self.target_network.predict(self._flatten_obs(next_obs), verbose = 0).max()
            tarnet_out[0][action] = q_new

            self.policy_network.fit(self._flatten_obs(obs), tarnet_out, verbose = 0)



    def train(self, episodes):
        rewards_per_episode = np.zeros(episodes) 

        for e in range(episodes):
            rewards_breakdown = [0, 0, 0, 0]
            obs, _ = self.env.reset()
            terminated = False
            truncated = False

            while (not terminated and not truncated):
                if np.random.random() > self.epsilon:
                    action = self.env.action_space.sample()
                else:
                    action = self.policy_network.predict(self._flatten_obs(obs), verbose = 0)[0].argmax()

                next_obs, reward, terminated, truncated, info = self.env.step(action)

                transition = (obs, reward, action, next_obs, terminated or truncated)
                self.memory.append(transition)

                obs = next_obs

                rewards_per_episode[e] += reward
                rewards_breakdown = np.add(rewards_breakdown, list(info["rewards"].values()))

            if len(self.memory) > self.mini_batch_size:
                mini_batch = random.sample(self.memory, self.mini_batch_size)
                self._optimize(mini_batch)

            self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)

            self._sync_network()

            print(f"Episode: {e + 1}")
            print(f"Epsilon: {self.epsilon}")
            print(utils.format_rewards(rewards_breakdown))

        return rewards_per_episode
    


    def act(self):
        rewards_breakdown = [0, 0, 0, 0]
        obs, _ = self.env.reset()
        terminated = False
        truncated = False

        while (not terminated and not truncated):
            action = self.target_network.predict(self._flatten_obs(obs), verbose = 0)[0].argmax()
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            obs = next_obs
            rewards_breakdown = np.add(rewards_breakdown, list(info["rewards"].values()))

        print(utils.format_rewards(rewards_breakdown))


                

###########################################################

LOAD_PRETRAINED = False
VERSION_NUM = "v1-0-1"
EPISODES_NUM = 500
POLICY_DIR = f"./models/dqn/dqn-{VERSION_NUM}-e{EPISODES_NUM}-policy.keras"
TARGET_DIR = f"./models/dqn/dqn-{VERSION_NUM}-e{EPISODES_NUM}-target.keras"

import utils
utils.init()
env = gym.make("WaterHeater-v0")
agent = DQNAgent(env, 6, 4)

if LOAD_PRETRAINED:
    agent.load_models(POLICY_DIR, TARGET_DIR)
    agent.act()
else:
    rewards = agent.train(EPISODES_NUM)
    print(rewards)

    agent.policy_network.save(POLICY_DIR)
    agent.target_network.save(TARGET_DIR)
