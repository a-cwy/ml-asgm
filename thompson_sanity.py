from exploration import ThompsonSampling
import numpy as np

ts = ThompsonSampling(
    4,
    1,
    1
)

dist = np.array([
    [10, 7],
    [5, 9],
    [8, 7],
    [13, 2]
])

total = 0

for _ in range(10000):
    action = ts.get_action()
    trues = np.random.normal(dist[:, 0], dist[:, 1], 4)
    reward = 1 if (action == trues.argmax()) else 0
    ts.update(action, reward)
    total += reward

print(dist)
print(ts.distributions[:, :2])
print(total)