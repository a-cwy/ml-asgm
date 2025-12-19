import numpy as np
import matplotlib.pyplot as plt

rewards = np.load("ppo_rewards.npy")

window = 20
moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')

plt.figure(figsize=(12,6))
plt.plot(rewards,label='Episode Reward',alpha=0.4)
plt.plot(range(window-1, len(rewards)), moving_avg, label=f"{window}-Episode Moving Average", linewidth=3)

plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("PPO Training Rewards per Episode")
plt.legend()
plt.grid(True)
plt.show()
