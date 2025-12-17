import utils
import exploration_results_dqn as xpl
import matplotlib.pyplot as plt
import numpy as np

def metrics():
    data = xpl.v5002().data

    print("\n")
    print(f"Final Reward: {data[-1]:.4f}")
    print(f"Cum. Sum: {np.sum(data):.4f}")
    print(f"Min: {min(data):.4f}")
    print(f"Max: {max(data):.4f}")
    print(f"Mean: {np.mean(data):.4f}")
    print(f"S.D.: {np.std(data):.4f}")
    print(f"Coef. Variation: {np.std(data)/np.mean(data):.4f}")
    print("\n")

def compare():
    cmp = [
        xpl.v5000().data,
        xpl.v5001().data,
        xpl.v5002().data
    ]

    plt.figure(figsize = (16, 8))
    plt.yscale('symlog')
    plt.plot(cmp[0], label = 'decay = 0.995')
    plt.plot(cmp[1], 'orange', label = 'decay = 0.975')
    plt.plot(cmp[2], 'green', label = 'decay = 0.95')
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Total Reward per Episode")
    plt.legend(loc = "center right")
    plt.grid()
    plt.show()

compare()