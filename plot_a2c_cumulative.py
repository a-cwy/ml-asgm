import numpy as np
import matplotlib.pyplot as plt

# Load rewards from a2c training
rewards1 = np.load("models/a2c/a2c-v1-e500-rewards.npy")
rewards2 = np.load("models/a2c/a2c-v2-e500-rewards.npy")
rewards3 = np.load("models/a2c/a2c-v3-e500-rewards.npy")


# Compute metrics over the entire reward series (no last-N/window logic)
def compute_metrics(rewards: np.ndarray) -> dict:
	r = np.asarray(rewards, dtype=np.float32)
	n = r.shape[0]

	mean_overall = float(np.mean(r)) if n > 0 else float("nan")
	std_overall = float(np.std(r)) if n > 1 else float("nan")

	# Coefficient of variation (std / mean) and a simple stability proxy (1/std)
	cv_overall = float(std_overall / mean_overall) if (mean_overall != 0 and not np.isnan(mean_overall)) else float("nan")
	stability = float(1.0 / std_overall) if (std_overall != 0 and not np.isnan(std_overall)) else float("nan")

	metrics = {
		"episodes": int(n),
		"final_reward": float(r[-1]) if n > 0 else float("nan"),
		"mean_overall": mean_overall,
		"std_overall": std_overall,
		"min": float(np.min(r)) if n > 0 else float("nan"),
		"max": float(np.max(r)) if n > 0 else float("nan"),
		"cumulative_reward": float(np.sum(r)) if n > 0 else float("nan"),
		"cv_overall": cv_overall,
		"stability": stability,
	}

	return metrics



def print_metrics(name: str, m: dict):
	print(f"=== {name} ===")
	print(
		f"Episodes: {m['episodes']} | Final: {m['final_reward']:.3f} | "
		f"Cum.Sum: {m['cumulative_reward']:.3f} | Min/Max: {m['min']:.3f}/{m['max']:.3f}"
	)
	print(
		f"Overall -> Mean: {m['mean_overall']:.3f}, Std: {m['std_overall']:.3f}, CV: {m['cv_overall']:.3f}, "
		f"Stability (1/std): {m['stability']:.3f}"
	)
	print("\n")


# Compute and print metrics for each setting
metrics1 = compute_metrics(rewards1)
metrics2 = compute_metrics(rewards2)
metrics3 = compute_metrics(rewards3)

print_metrics("learning_rate = 1e-3", metrics1)
print_metrics("learning_rate = 1e-4", metrics2)
print_metrics("learning_rate = 3e-4", metrics3)

# Plot raw episode rewards
plt.figure(figsize=(12, 6))
plt.plot(np.cumsum(rewards1), label="learning_rate = 1e-3", linewidth=2)
plt.plot(np.cumsum(rewards2), label="learning_rate = 1e-4", linewidth=2)
plt.plot(np.cumsum(rewards3), label="learning_rate = 3e-4", linewidth=2)

plt.xlabel("Episode")
plt.ylabel("Cumulative Reward")
plt.title("Cumulative Reward Across A2C Learning Rates")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
plt.close()
