PARAMETER = "n_steps"
PARAM_VALUES = [20,150,100]

import numpy as np
import matplotlib.pyplot as plt

# Load rewards from a2c training
rewards1 = np.load("models/a2c/a2c-v4-e100-rewards.npy")
rewards2 = np.load("models/a2c/a2c-v5-e100-rewards.npy")
rewards3 = np.load("models/a2c/a2c-v6-e100-rewards.npy")


# Compute metrics over the entire reward series
def compute_metrics(rewards: np.ndarray) -> dict:
	r = np.asarray(rewards, dtype=np.float32)
	n = r.shape[0]

	mean_overall = float(np.mean(r)) if n > 0 else float("nan")
	std_overall = float(np.std(r)) if n > 1 else float("nan")

	# Coefficient of variation (std / mean) and a simple stability proxy (1/std)
	cv_overall = float(std_overall / mean_overall) if (mean_overall != 0 and not np.isnan(mean_overall)) else float("nan")

	metrics = {
		"episodes": int(n),
		"final_reward": float(r[-1]) if n > 0 else float("nan"),
		"mean_overall": mean_overall,
		"std_overall": std_overall,
		"min": float(np.min(r)) if n > 0 else float("nan"),
		"max": float(np.max(r)) if n > 0 else float("nan"),
		"cumulative_reward": float(np.sum(r)) if n > 0 else float("nan"),
		"cv_overall": cv_overall
	}

	return metrics



def print_metrics(name: str, m: dict):
	print(f"=== {name} ===")
	print(
		f"Episodes: {m['episodes']} | Final: {m['final_reward']:.3f} | "
		f"Cum.Sum: {m['cumulative_reward']:.3f} | Min/Max: {m['min']:.3f}/{m['max']:.3f}"
	)
	print(
		f"Overall -> Mean: {m['mean_overall']:.3f}, Std: {m['std_overall']:.3f}, CV: {m['cv_overall']:.3f} "
	)
	print("\n")


# Compute and print metrics for each setting
metrics1 = compute_metrics(rewards1)
metrics2 = compute_metrics(rewards2)
metrics3 = compute_metrics(rewards3)

print_metrics(f"{PARAMETER} = {PARAM_VALUES[0]}", metrics1)
print_metrics(f"{PARAMETER} = {PARAM_VALUES[1]}", metrics2)
print_metrics(f"{PARAMETER} = {PARAM_VALUES[2]}", metrics3)

# Plot raw episode rewards
plt.figure(figsize=(12, 6))
plt.plot(rewards1, label=f"{PARAMETER} = {PARAM_VALUES[0]}", linewidth=2)
plt.plot(rewards2, label=f"{PARAMETER} = {PARAM_VALUES[1]}", linewidth=2)
plt.plot(rewards3, label=f"{PARAMETER} = {PARAM_VALUES[2]}", linewidth=2)

plt.xlabel("Episode")
plt.yscale("symlog")
plt.ylabel("Reward (log y)")
plt.title(f"Reward Across A2C {PARAMETER}")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"./models/a2c/a2c {PARAMETER} comp.png", dpi=200)
plt.show()
plt.close()
