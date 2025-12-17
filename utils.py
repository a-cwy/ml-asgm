from environment import WaterHeaterEnv
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

def init():
    """
    Initializes project space.
    Registers environment into gymnasium to be accessed using gym.make()

    Returns:
        None
    """
    gym.register(
        id = "WaterHeater-v0",
        entry_point = WaterHeaterEnv,
        max_episode_steps = 2 * 7 * 96
    )

def format_rewards(reward_breakdown):
    formatted_string = f"""Rewards Breakdown
    Comfort : {reward_breakdown[0]}
    Hygiene : {reward_breakdown[1]}
    Energy  : {reward_breakdown[2]}
    Safety  : {reward_breakdown[3]}
    Total   : {sum(reward_breakdown)}
    """

    return formatted_string

def plot_breakdown_cumulative(reward_breakdown):
    cumsum = np.swapaxes(np.cumsum(reward_breakdown, axis = 0), 0, 1)
    total = np.sum(cumsum, axis = 0)

    plt.plot(cumsum[0], 'g-', label = 'Comfort') # Comfort
    plt.plot(cumsum[1], 'b-', label = 'Hygiene') # Hygiene
    plt.plot(cumsum[2], 'y-', label = 'Energy') # Energy
    plt.plot(cumsum[3], 'r-', label = 'Safety') # Safety
    plt.plot(total, 'k-', label = 'Total Reward') # Total
    plt.legend(loc = "upper left")
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.title("Cumulative Rewards by Type")
    plt.grid()
    plt.show()

def plot_rewards(rewards_per_episode):
    plt.plot(rewards_per_episode)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Total Reward per Episode")
    plt.grid()
    plt.savefig(path, dpi=200)
    plt.show()
    plt.close()