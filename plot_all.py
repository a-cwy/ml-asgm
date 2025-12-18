"""
To compare different RL algorithms (PPO, A2C, SAC, DQN) based on their ACT episodic rewards
"""
VERSION = "v1"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

rewards_rulebased = np.load("models/rulebased_rewards.npy")
rewards_ppo = np.load("models/ppo/ppo_act_100_rewards.npy")
rewards_a2c = np.load("models/a2c/a2c_act100_total_rewards.npy")
rewards_sac = np.load("models/sac/new_sac_act_100_rewards.npy")
rewards_dqn = [1048.309117729268, 1069.5026737583723, 1048.4517957213461, 1045.1537590423807, 936.5585172348857, 1059.5782669659077, 935.7592384355157, 1058.9103801163906, 935.1753712722582, 953.8739630154133, 932.8458162373181, 933.3430721550471, 1061.793842397295, 1031.0737131902658, 1046.0178865213454, 912.9244045774146, 960.5409293446482, 944.5169216905992, 1045.864845331248, 1069.1155458495336, 1064.3340194190837, 984.7504184760563, 1059.0033365161835, 965.9267200111822, 1032.4666642148488, 1067.5829893581765, 1036.1908623539887, 1060.5260430810517, 989.967596410525, 1027.1484758403258, 1058.2826729333992, 964.7959256028282, 995.8982007826423, 859.4792688166358, 978.7717133366386, 985.1540525668643, 961.7732568574702, 948.910449049906, 1070.5902417760917, 996.1722966933021, 1067.310318543259, 997.2504223353833, 1063.4375834392026, 951.4783538162411, 1043.801400660975, 1047.8041027954882, 1050.0716706400779, 1061.0095775788368, 1065.5119396175705, 989.5010221069944, 959.7574154870041, 927.3205011227878, 1060.4078402765854, 1067.1121461065816, 922.9327010647459, 1065.8477143554014, 983.4521041631646, 1030.6229856361451, 1035.3422715636052, 962.9829747429824, 1027.6121784441928, 952.5443330394389, 1050.7757515922654, 959.7503709844952, 915.9488276544404, 1066.0145190713945, 1056.6618038826477, 921.85843460079, 961.4564152118644, 1031.7146663650765, 959.186712278045, 1059.9325759213684, 928.6741942500038, 1052.3583227241413, 934.870040082888, 966.9816806336692, 1049.4340249374661, 1070.1417709288212, 1068.7304387326324, 944.5920860646308, 997.9253792596041, 988.6810989944843, 935.216399937951, 912.7127437136264, 1065.4309612583234, 1058.4786736271524, 972.114503551018, 933.8915369617293, 972.6770515153546, 1068.9428306430211, 981.7287299330659, 917.8189735540859, 908.9349553264358, 920.7866156161895, 990.1941942713624, 980.7409544968231, 1043.3269687561724, 1029.776316100874, 1036.2231716647805, 959.3463735813067]


clrmap = plt.get_cmap('rainbow')

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
    from brokenaxes import brokenaxes
    from matplotlib.gridspec import GridSpec

    # compute metrics
    avgs = np.array([np.mean(r) for r in reward_lists])
    cums = np.array([np.sum(r) for r in reward_lists])
    variance = np.array([np.var(r) for r in reward_lists])

    x = np.arange(len(alg_names))
    width = 0.6

    # Create figure and GridSpec with adjusted spacing
    fig = plt.figure(figsize=(14, 5))
    gs = GridSpec(1, 2, figure=fig, wspace=0.35)  # Increase horizontal spacing
    fig.suptitle('RL Model Comparison Metrics (Evaluated over 100 episodes)', y=0.98)

    # get a colormap and generate distinct colors for each algorithm
    cmap = clrmap
    colors = cmap(np.linspace(0, 1, len(alg_names)))

    # 1 . average
    # Create brokenaxes for first subplot using GridSpec (there should be 2 axes)
    # bax1 = brokenaxes(
    #     ylims=((-35000, -33000), (-1500, 1500)), 
    #     subplot_spec=gs[0],
    #     fig=fig,
    #     despine=False  
    # )
    # bax1.bar(x, avgs, width, color=colors)
    # bax1.set_title('Average Episodic Reward', pad=10)
    # bax1.set_xticks(x)
    # bax1.set_xticklabels(alg_names, rotation=45, ha='right')
    # for ax in bax1.axs:
    #     ax.grid(axis='y', linestyle='--', alpha=0.4)
    #     ax.set_xticks(x)  
    #     ax.set_xticklabels(alg_names, rotation=45, ha='right')

    ax1 = fig.add_subplot(gs[0])
    ax1.bar(x, avgs, width, color=colors)
    ax1.set_title('Average Episodic Reward', pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(alg_names, rotation=45, ha='right')
    ax1.grid(axis='y', linestyle='--', alpha=0.4)

    # # 2 . cumulative
    # bax2 = brokenaxes(
    #     ylims=((-3.5e6, -3.3e6), (-1.5e5, 1.2e5)), 
    #     subplot_spec=gs[1],
    #     fig=fig,
    #     despine=False 
    # )
    # ax2 = fig.add_subplot(gs[1])
    # ax2.bar(x, cums, width, color=colors)
    # ax2.set_xticks(x)
    # ax2.set_xticklabels(alg_names, rotation=45, ha='right')
    # ax2.grid(axis='y', linestyle='--', alpha=0.4)
    # bax2.bar(x, cums, width, color=colors)
    # bax2.set_title('Cumulative Reward', pad=10)
    # bax2.set_xticks(x)  
    # bax2.set_xticklabels(alg_names, rotation=45, ha='right')
    # for ax in bax2.axs:
    #     ax.grid(axis='y', linestyle='--', alpha=0.4)
    #     ax.set_xticks(x)  
    #     ax.set_xticklabels(alg_names, rotation=45, ha='right')
    #     if ax == bax2.axs[1]:
    #         ax.set_yticks([-3.5e6, -3.4e6, -3.3e6,], labels=['-3500000', '-3400000', '-3300000', ])
    
    ax3 = fig.add_subplot(gs[1])
    ax3.bar(x, variance, width, color=colors)
    ax3.set_title('Performance Stability (Variance)', pad=10)
    ax3.set_ylabel('Variance (log scale)')
    ax3.set_xticks(x)
    ax3.set_xticklabels(alg_names, rotation=45, ha='right')
    ax3.set_yscale("log")
    ax3.grid(axis='y', linestyle='--', alpha=0.4)

    print(f"Average: {avgs}, \n Cumulative: {cums},\n Variance: {variance}")
    
    # Adjust layout with custom parameters
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
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
        # if(name == "SAC"):
        #     continue
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