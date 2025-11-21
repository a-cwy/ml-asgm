# Minimum required imports, include in each agent file
import utils
import gymnasium as gym
import numpy as np

# Always run this function first. Registers the env into gymnasium and makes gym.make("WaterHeater-v0") possible
utils.init()

# Initializing the environment and running a reset to randomize starting state.
env = gym.make("WaterHeater-v0")
obs, _ = env.reset()
print(obs)

total_reward = 0.0
reward_breakdown = [0.0, 0.0, 0.0, 0.0]

# Perform first action randomly to get variables.
next_obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
hygiene = False

# Rule-based agent to meet basic needs. Maximum (365 * 96) iterations per episode.
for i in range(365 * 96):
    if i % 96 == 0: hygiene = True # Cycle to 70 every day to prevent hygiene punishemnt. 
    if next_obs["waterTemperature"] > 70: hygiene = False

    # Keep temperature around 64C
    if next_obs["waterTemperature"] < 64 or hygiene:
        next_obs, reward, terminated, truncated, info = env.step(3)
    else:
        next_obs, reward, terminated, truncated, info = env.step(0)

    total_reward += np.float32(reward)

    # Cumulative reward for each subcategory.
    reward_breakdown[0] += info["rewards"]["comfort"]
    reward_breakdown[1] += info["rewards"]["hygiene"]
    reward_breakdown[2] += info["rewards"]["energy"]
    reward_breakdown[3] += info["rewards"]["safety"]

    if terminated or truncated:
        print("Simulation ended.")
        break

# Utility function to print a breakdown of the rewards.
print(utils.format_rewards(reward_breakdown))
print(next_obs)
print(info)