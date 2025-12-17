"""
To compare different RL algorithms (PPO, A2C, SAC, DQN) based on their episodic rewards
"""
VERSION = "v1"

import pandas as pd
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
rewards_dqn = scores = [935.0,839.8675821958195,-46599.28104512051,-28401.67552375052,-12771.431145987735,-11781.923217418109,-12466.87690406844,-2534.5575649473676,-1923.8081616785134,-1251.7647329710755,-1402.9728665766509,-1500.288597696093,-1354.9777266953047,-1386.1129186518099,-1705.071564531073,-1483.8735556503304,-875.5743795644739,894.652987431722,1007.5,902.5730616101582,895.2700183834102,1016.25,989.7332857515129,947.75,991.75,922.9129623896808,854.5566583479724,761.5145158218805,803.1305180583746,783.6797951608809,771.75,956.3997982320761,961.25,917.3105234939761,827.75,913.5,973.2764455117385,430.1037305781626,906.75,792.3028778332714,995.1443068169056,978.166913583245,897.0081595168251,983.75,953.0,989.2266002676779,999.0,935.25,988.5,989.25,995.25,1004.5,932.75,987.5,830.75,841.75,918.25,990.0,-1999.203003001121,884.5,964.5,1009.0,984.5,982.25,995.25,991.75,904.25,998.75,986.0,946.5,916.25,855.25,961.0,977.5,995.5,932.25,989.25,967.5,961.0,937.5,991.5,934.5,962.0,938.142626501628,962.75,957.25,-22750.345161314995,-50231.54299036813,-3690.6514743138764,941.25,1073.25,921.75,923.0,903.5,1009.0,1052.25,1027.5751020860762,1056.5,1021.5687557059748,1061.5]

clrmap = plt.get_cmap('rainbow')

rewards_rulebased = np.load("models/rulebased_rewards.npy")

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

def compute_metrics(rewards):
    r = np.asarray(rewards, dtype=np.float32)
    n = r.shape[0]

    mean_overall = float(np.mean(r)) if n > 0 else float("nan")
    std_overall = float(np.std(r)) if n > 1 else float("nan")

    cv_overall = float(std_overall / mean_overall) if (mean_overall != 0 and not np.isnan(mean_overall)) else float("nan")

    metrics = {
        "episodes": int(n),
        "final_reward": float(r[-1]) if n > 0 else float("nan"),
        "mean_overall": mean_overall,
        "std_overall": std_overall,
        "min": float(np.min(r)) if n > 0 else float("nan"),
        "max": float(np.max(r)) if n > 0 else float("nan"),
        "cumulative_reward": float(np.sum(r)) if n > 0 else float("nan"),
        "cv_overall": cv_overall,
    }

    return metrics

metrics_list = [compute_metrics(r) for r in reward_lists]
rows = []
for name, m in zip(alg_names, metrics_list):
    row = {"algorithm": name}
    row.update(m)
    rows.append(row)

df_metrics = pd.DataFrame(rows)
cols = ["algorithm", "episodes", "final_reward", "mean_overall", "std_overall", "min", "max", "cumulative_reward", "cv_overall"]
df_metrics = df_metrics[cols]

print("Metrics table:")
print(df_metrics.to_string(index=False))

# Save table to CSV for report
df_metrics.to_csv(f"plots/metrics_table_{VERSION}.csv", index=False) ; print(f"Saved metrics to: plots/metrics_table_{VERSION}.csv")

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
    axs[2].set_yscale("log") #Comment this line to use linear scale

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
    else:
        yscale = 'linear'

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