"""
To compare different RL algorithms (PPO, A2C, SAC, DQN) based on their episodic rewards
"""


import numpy as np
import matplotlib.pyplot as plt

rewards_ppo = np.load("models/ppo/ppo_rewards.npy")
rewards_a2c = np.load("models/a2c/a2c-v4-e100-rewards.npy")
rewards_sac = [
    -2759.0, -2734.2, -2734.0, -2734.9, -2765.8, -2691.0, -2687.2, -2825.9, -2758.2, -2687.6,
    -2739.4, -2691.1, -2758.0, -2705.7, -2797.3, -2705.5, -2719.4, -2683.0, -2683.0, -2781.9,
    -2700.2, -2697.7, -2762.0, -2708.7, -2750.0, -2740.8, -2733.3, -2683.0, -2690.7, -2763.7,
    -2733.2, -2684.8, -2737.4, -2685.4, -2752.4, -2683.0, -2683.0, -2773.5, -2712.6, -2683.0,
    -2683.0, -2715.2, -2684.3, -2727.3, -2742.3, -2706.3, -2700.9, -2686.8, -2741.8, -2750.3,
    -2717.8, -2721.2, -2692.8, -2688.0, -2778.9, -2803.4, -2683.8, -2699.6, -2781.5, -2686.5,
    -2805.1, -2684.4, -2742.5, -2686.5, -2687.8, -2744.0, -2683.6, -2685.9, -2683.0, -2787.0,
    -2684.7, -2683.0, -2809.0, -2751.2, -2685.6, -2763.1, -2784.3, -2718.6, -2748.1, -2688.3,
    -2751.5, -2738.7, -2686.8, -2683.0, -2709.2, -2698.1, -2683.9, -2817.8, -2683.0, -2693.4,
    -2683.0, -2692.7, -2819.5, -2693.4, -2695.8, -2695.3, -2686.9, -2690.8, -2683.0, -2794.0
]
rewards_dqn = scores = [
    539.37078651, -3481.63863318, -3639.25973046, -2745.69394765,
    -374.81397261, -1104.12377086, -4041.55744872, -6040.94734777,
    -9453.85118032, -2177.62476291, -900.60743569, -2406.21102496,
    928.5, 939.0, 650.0, 915.25,
    570.82786021, 551.03056535, 570.15384196, 932.5,
    732.11328902, 211.9664788, 929.75, 752.21994982,
    933.5, 923.0, 367.91656733, -1280.15619659,
    -1310.70555837, -2379.87262609, -2180.13565058, -4194.93367662,
    -9194.33190726, 722.89705036, 178.13276659, 921.25,
    909.5, 900.0, 899.0, 739.59907052,
    -343.13598943, -904.8877411, 917.5, 928.0,
    922.25, 924.5, 915.25, 943.25,
    457.5, 675.75, 923.25, 934.75,
    936.5, 928.25, -508.25, 914.0,
    720.75, 942.5, 808.75, -2805.40354206,
    -14118.21196109, -8772.96495772, -22897.85332788, 176.02520818,
    -555.41101977, -22657.39489829, 937.75, 940.5,
    944.0, 955.5, 962.75, 949.25,
    954.0, 968.5, 818.5, 974.5,
    643.75, 777.25, 986.75, 979.25,
    987.5, 971.25, 724.25, 685.25,
    -363.75, 997.75, -1937.5, 997.75,
    -3083.5, 959.25, 957.0, 961.0,
    978.5, 973.0, 964.5, 948.0,
    977.5, 988.0, 938.0, 925.5
]

clrmap = plt.get_cmap('rainbow')

rewards_rulebased = np.load("rulebased_rewards.npy")

print(len(rewards_ppo), len(rewards_a2c), len(rewards_sac), len(rewards_dqn), len(rewards_rulebased))

# Ensure all reward lists are of the same length by truncating to the shortest
if len(rewards_ppo) != len(rewards_a2c) or len(rewards_ppo) != len(rewards_sac) or len(rewards_ppo) != len(rewards_dqn) or len(rewards_rulebased) != len(rewards_ppo):
    min_length = min(len(rewards_ppo), len(rewards_a2c), len(rewards_sac), len(rewards_dqn), len(rewards_rulebased))
    rewards_ppo = rewards_ppo[:min_length]
    rewards_a2c = rewards_a2c[:min_length]
    rewards_sac = rewards_sac[:min_length]
    rewards_dqn = rewards_dqn[:min_length]
    rewards_rulebased = rewards_rulebased[:min_length]
    print(f"Truncated all reward lists to length: {min_length}")

alg_names = ["PPO", "A2C", "SAC", "DQN", "Rule-based"]
reward_lists = [rewards_ppo, rewards_a2c, np.array(rewards_sac), np.array(rewards_dqn), rewards_rulebased]
def plot_comparison(alg_names, reward_lists, save_path="plots/comparison_metrics.png"):
    import matplotlib.pyplot as plt

    # compute metrics
    avgs = np.array([np.mean(r) for r in reward_lists])
    cums = np.array([np.sum(r) for r in reward_lists])
    variance = np.array([np.var(r) for r in reward_lists])

    x = np.arange(len(alg_names))
    width = 0.6

    fig, axs = plt.subplots(1, 3, figsize=(14, 5))
    plt.suptitle('RL Model Comparison Metrics (Evaluated over 100 episodes)')

    # get a colormap and generate distinct colors for each algorithm
    cmap = clrmap
    colors = cmap(np.linspace(0, 1, len(alg_names)))

    axs[0].bar(x, avgs, width, color=colors)
    axs[0].set_title('Average Episodic Reward')
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(alg_names, rotation=45, ha='right')

    axs[1].bar(x, cums, width, color=colors)
    axs[1].set_title('Cumulative Reward')
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(alg_names, rotation=45, ha='right')

    axs[2].bar(x, variance, width, color=colors)
    axs[2].set_title('Learning Instability (Variance)')
    axs[2].set_ylabel('Variance (log scale)')
    axs[2].set_xticks(x)
    axs[2].set_xticklabels(alg_names, rotation=45, ha='right')
    axs[2].set_yscale("log")

    for ax in axs:
        ax.grid(axis='y', linestyle='--', alpha=0.4)

    # print(type(avgs))
    print(f"Average: {avgs}, \n Cumulative: {cums},\n Variance: {variance}")
    plt.tight_layout()
    fig.savefig(save_path, dpi=200)
    print(f"Saved comparison plot to: {save_path}")
    plt.show()


# Prepare and plot using available reward arrays

plot_comparison(alg_names=alg_names, reward_lists=reward_lists, save_path="plots/comparison_metrics.png")


def plot_episodic_line(alg_names, reward_lists, save_path="plots/episodic_rewards.png", cumulative=False, logscale=False, linthresh=1.0):
    """Plot episodic rewards as lines for each algorithm and save the figure.

    If `cumulative` is True, plot cumulative rewards (np.cumsum) per episode.
    If `logscale` is True, use `log` y-scale when all series are positive,
    otherwise use `symlog` which supports negative values (controlled by `linthresh`).
    """
    plt.figure(figsize=(10, 6))
    cmap = clrmap
    colors = cmap(np.linspace(0, 1, len(alg_names)))

    processed = []
    # zip together names, rewards, colors
    for name, rewards, color in zip(alg_names, reward_lists, colors):
        r = np.array(rewards)
        if cumulative:
            r = np.cumsum(r)
        processed.append((name, r, color))

    # decide y-scale
    if logscale:
        all_positive = all((series > 0).all() for _, series, _ in processed)
        yscale = 'log' if all_positive else 'symlog'
    # else:
    #     yscale = 'linear'

    for name, r, color in processed:
        episodes = np.arange(1, len(r) + 1)
        plt.plot(episodes, r, label=name, color=color, linewidth=1.5)

    plt.xlabel('Episode')
    plt.ylabel(('Cumulative Reward' if cumulative else 'Reward') + (' (log scale)' if logscale else ''))
    plt.title(( 'Cumulative Reward per Episode' if cumulative else 'Episodic Rewards') + ( ' [log]' if logscale else ''))
    plt.legend(loc='best')
    plt.grid(linestyle='--', alpha=0.3)

    if yscale == 'log':
        plt.yscale('log')
    elif yscale == 'symlog':
        plt.yscale('symlog', linthresh=linthresh)
    plt.tight_layout()

    # adjust default filename when plotting cumulative or logscale variants
    if save_path == "plots/episodic_rewards.png":
        if cumulative and logscale:
            save_path = "plots/episodic_cumulative_rewards_log.png"
        elif cumulative:
            save_path = "plots/episodic_cumulative_rewards.png"
        elif logscale:
            save_path = "plots/episodic_rewards_log.png"
    plt.savefig(save_path, dpi=200)
    print(f"Saved episodic reward plot to: {save_path} (yscale={yscale})")
    plt.show()


# default: episodic (non-cumulative, linear)
plot_episodic_line(alg_names, reward_lists, logscale=True)