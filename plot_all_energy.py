import numpy as np
import matplotlib.pyplot as plt
import utils

    # plt.plot(cumsum[0], 'g-', label = 'Comfort') # Comfort
    # plt.plot(cumsum[1], 'b-', label = 'Hygiene') # Hygiene
    # plt.plot(cumsum[2], 'y-', label = 'Energy') # Energy
    # plt.plot(cumsum[3], 'r-', label = 'Safety') # Safety
reward_types = ['Comfort', 'Hygiene', 'Energy', 'Safety']

def plot_all_rewards():
    # Load data
    a2c = np.load("models/a2c/a2c_1_episode_rewards_breakdown.npy")
    rulebased = np.load("models/rulebased_1_episode_rewards_breakdown.npy")
    dqn = np.load("models/dqn/dqn_v5-0-0-0_episode_rewards_breakdown.npy")
    ppo = np.load("models/ppo/ppo_eval_step_rewards_breakdown.npy")
    sac = np.load("models/sac/sac_1_episode_rewards.npy")

    # Truncate to minimum length
    min_len = min(a2c.shape[0], rulebased.shape[0], dqn.shape[0], ppo.shape[0], sac.shape[0])
    a2c = a2c[:min_len]
    rulebased = rulebased[:min_len]
    dqn = dqn[:min_len]
    ppo = ppo[:min_len]
    sac = sac[:min_len]

    # Create 2x2 subplots
    # fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    plt.figure(figsize=(15, 10))
    # # Flatten axes for easier iteration
    # axes = axes.flatten()
    
    # Plot each reward type
    for idx, reward_type in enumerate(reward_types):
        # Skip the first row which is all zeros and extract the specific reward
        a2c_reward = np.cumsum(a2c[1:, idx].flatten())
        rulebased_reward = np.cumsum(rulebased[1:, idx].flatten())
        dqn_reward = np.cumsum(dqn[1:, idx].flatten())
        ppo_reward = np.cumsum(ppo[1:, idx].flatten())
        sac_reward = np.cumsum(sac[1:, idx].flatten())
        
        # Plot on the corresponding subplot
        plt.subplot(2, 2, idx + 1)
        plt.plot(rulebased_reward, 'r--', label='Rule-based', linewidth=1.5)
        plt.plot(a2c_reward, 'y-', label='A2C', linewidth=1.5)
        plt.plot(dqn_reward, 'b-', label='DQN', linewidth=1.5)
        plt.plot(ppo_reward, 'g-', label='PPO', linewidth=1.5)
        plt.plot(sac_reward, 'm-', label='SAC', linewidth=1.5)
        
        plt.xlabel("Step", fontsize=11)
        plt.ylabel(f"{reward_type} Reward", fontsize=11)
        plt.title(f"Cumulative {reward_type} Reward", fontsize=12)
        plt.legend(loc="best", fontsize=9)
        plt.grid(True, alpha=0.3)
    
    plt.suptitle('Cumulative Rewards per Step (1 Episode - 1344 Steps)', fontsize=16, y=0.98, fontweight='bold')
    plt.tight_layout()
    plt.savefig("plots/cum_1_episode_rewards_subplots.png", dpi=300, bbox_inches='tight')
    plt.show()

plot_all_rewards()