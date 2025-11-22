import utils
import gymnasium as gym
import numpy as np
import random
from dqn import DQNAgent

utils.init()

env = gym.make("WaterHeater-v0")
agent = DQNAgent(env, state_size = 6, action_size = 4)

EPISODES = 100
                 
for e in range(EPISODES):
    obs, info = agent.env.reset()
    total_reward = 0.0

    for i in range(2000):
        action = agent.act(np.array(list(obs.values())).reshape(-1, 6))
        next_obs, reward, terminated, truncated, info = agent.env.step(action)
        agent.remember(
            np.array(list(obs.values())).reshape(-1, 6), 
            action, 
            reward, 
            np.array(list(next_obs.values())).reshape(-1, 6), 
            terminated or truncated
        )
        total_reward += reward # type: ignore
        obs = next_obs

    if len(agent.memory) > 128:
        agent.replay(batch_size = 128)

    print(f"Episode: {e+1}")
    print(f"Reward: {total_reward:.2f}")
    print(f"Epsilon: {agent.epsilon:.5f}\n")