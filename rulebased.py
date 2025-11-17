import utils
import gymnasium as gym
import numpy as np

utils.init()

env = gym.make("WaterHeater-v0")
obs, _ = env.reset()
print(obs)

total_reward = 0.0
reward_breakdown = [0.0, 0.0, 0.0, 0.0, 0.0]

next_obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
hygiene = False
for i in range(365 * 96):
    if i % 96 == 0: hygiene = True
    if next_obs["waterTemperature"] > 70: hygiene = False

    if next_obs["waterTemperature"] < 64 or hygiene:
        next_obs, reward, terminated, truncated, info = env.step(1)
    else:
        next_obs, reward, terminated, truncated, info = env.step(0)

    total_reward += reward

    reward_breakdown[0] += info["rewards"]["comfort"]
    reward_breakdown[1] += info["rewards"]["hygiene"]
    reward_breakdown[2] += info["rewards"]["energy"]
    reward_breakdown[3] += info["rewards"]["safety"]

    if terminated or truncated:
        print("Simulation ended.")
        break

print(utils.format_rewards(reward_breakdown))