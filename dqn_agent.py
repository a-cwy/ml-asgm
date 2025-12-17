import utils
import gymnasium as gym
from dqn_tensorflow import DQNAgent
from exploration import EpsilonGreedy, ThompsonSampling

LOAD_PRETRAINED = True
SAVE = False


utils.init()
env = gym.make("WaterHeater-v0")
param_list = [
    {
        "version_num": "v5-0-0-0",
        "learning_rate": 0.001,
        "discount_factor": 0.9,
        "epsilon_decay": 0.995
    },
    {
        "version_num": "v5-0-0-1",
        "learning_rate": 0.001,
        "discount_factor": 0.9,
        "epsilon_decay": 0.975
    },
    {
        "version_num": "v5-0-0-2",
        "learning_rate": 0.001,
        "discount_factor": 0.9,
        "epsilon_decay": 0.95
    },
    {
        "version_num": "v5-0-1-0",
        "learning_rate": 0.001,
        "discount_factor": 0.85,
        "epsilon_decay": 0.995
    },
    {
        "version_num": "v5-0-2-0",
        "learning_rate": 0.001,
        "discount_factor": 0.8,
        "epsilon_decay": 0.995
    },
    {
        "version_num": "v5-1-0-0",
        "learning_rate": 0.0001,
        "discount_factor": 0.9,
        "epsilon_decay": 0.995
    },
    {
        "version_num": "v5-1-1-0",
        "learning_rate": 0.0001,
        "discount_factor": 0.85,
        "epsilon_decay": 0.995
    },
    {
        "version_num": "v5-1-2-0",
        "learning_rate": 0.0001,
        "discount_factor": 0.8,
        "epsilon_decay": 0.995
    },
    {
        "version_num": "v5-2-0-0",
        "learning_rate": 0.00001,
        "discount_factor": 0.9,
        "epsilon_decay": 0.995
    },
    {
        "version_num": "v5-2-1-0",
        "learning_rate": 0.00001,
        "discount_factor": 0.85,
        "epsilon_decay": 0.995
    },
    {
        "version_num": "v5-2-2-0",
        "learning_rate": 0.00001,
        "discount_factor": 0.8,
        "epsilon_decay": 0.995
    }
]


VERSION_NUM = "v5-0-0-0"
EPISODES_NUM = 100
POLICY_DIR = f"./models/dqn/dqn-{VERSION_NUM}-e{EPISODES_NUM}-policy.keras"
TARGET_DIR = f"./models/dqn/dqn-{VERSION_NUM}-e{EPISODES_NUM}-target.keras"
PLOT_DIR = f"./models/dqn/tuning/{VERSION_NUM}.png"

agent = DQNAgent(
    env = env, 
    input_size = 6, 
    output_size = 4, 
    learning_rate = 0.001,
    discount_factor = 0.9,
    exploration = EpsilonGreedy(
        epsilon = 1, 
        min_epsilon = 0.01, 
        epsilon_decay = 0.995
    )
)

if LOAD_PRETRAINED:
    agent.load_models(POLICY_DIR, TARGET_DIR)
    tuned = agent.act()

    utils.plot_breakdown_cumulative(tuned)
else:
    rewards = agent.train(EPISODES_NUM)
    print(rewards)

    with open('models/dqn/tuning/temp.txt','a') as f:
        f.write(f'[{",".join(str(x) for x in rewards)}]\n')


    if SAVE:
        agent.policy_network.save(POLICY_DIR)
        agent.target_network.save(TARGET_DIR)