import numpy as np
import matplotlib.pyplot as plt
import utils

    # plt.plot(cumsum[0], 'g-', label = 'Comfort') # Comfort
    # plt.plot(cumsum[1], 'b-', label = 'Hygiene') # Hygiene
    # plt.plot(cumsum[2], 'y-', label = 'Energy') # Energy
    # plt.plot(cumsum[3], 'r-', label = 'Safety') # Safety


a2c = np.load("models/a2c/a2c_1_episode_rewards_breakdown.npy")
rulebased = np.load("models/rulebased_1_episode_rewards_breakdown.npy")

print(a2c.shape, rulebased.shape)

#skip the first row which is all zeros
a2c_energy = a2c[1:,2].flatten()
a2c_energy = np.cumsum(a2c_energy)

rulebased_energy = rulebased[1:,2].flatten()
rulebased_energy = np.cumsum(rulebased_energy)

plt.plot(rulebased_energy, 'r--', label = 'Rule-based') # Energy
plt.plot(a2c_energy, 'y-', label = 'A2C') # Energy
# a2c_comfort = a2c[:,0].flatten()
# plt.plot(a2c_comfort, 'g-', label = 'Comfort') # Comfort
plt.xlabel("Step")
plt.ylabel("Energy Reward")
plt.legend(loc = "best")
plt.title("Cummulative Energy Reward per Step (1 Episode - 1344 Steps)")
plt.grid()
plt.show()