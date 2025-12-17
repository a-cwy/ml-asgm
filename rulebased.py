# Minimum required imports, include in each agent file
import utils
import gymnasium as gym
import numpy as np

# Always run this function first. Registers the env into gymnasium and makes gym.make("WaterHeater-v0") possible
utils.init()

# Initializing the environment and running a reset to randomize starting state.
env = gym.make("WaterHeater-v0")

EPISODES = 100
reward_list = []
reward_breakdown = [0.0, 0.0, 0.0, 0.0]

#TO PLOT CUMMULATIVE REWARDS BREAKDOWN
rewards_breakdown_per_episode = [[0.0,0.0,0.0,0.0]]
# Rule-based agent to meet basic needs.
for e in range(EPISODES):
    obs, _ = env.reset()
    hygiene = False
    total_episode_reward = 0.0

    for i in range(365 * 96):
        if i % 96 == 0: hygiene = True # Cycle to 70 every day to prevent hygiene punishemnt. 
        if obs["waterTemperature"] > 70: hygiene = False

        # Keep temperature around 64C
        if obs["waterTemperature"] < obs["targetTemperature"] or hygiene:
            next_obs, reward, terminated, truncated, info = env.step(3)
        else:
            next_obs, reward, terminated, truncated, info = env.step(0)

        obs = next_obs
        rewards_breakdown_per_episode.append(list(info["rewards"].values()))
        total_episode_reward += np.float32(reward)

        # Cumulative reward for each subcategory.
        reward_breakdown[0] += info["rewards"]["comfort"]
        reward_breakdown[1] += info["rewards"]["hygiene"]
        reward_breakdown[2] += info["rewards"]["energy"]
        reward_breakdown[3] += info["rewards"]["safety"]

        if terminated or truncated:
            # print("Simulation ended.")
            break
    
    rewards_breakdown_per_episode.append(list(info["rewards"].values()))
    reward_list.append(total_episode_reward)

# Utility function to print a breakdown of the rewards.
reward_breakdown = np.divide(reward_breakdown, EPISODES)
print(utils.format_rewards(reward_breakdown))
# print("Reward list:", reward_list)
utils.plot_rewards(reward_list)
np.save("rulebased_rewards.npy", np.array(reward_list))

# print("Reward breakdown per episode:", rewards_breakdown_per_episode)
utils.plot_breakdown_cumulative(np.array(rewards_breakdown_per_episode))