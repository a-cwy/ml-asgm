import utils
import gymnasium as gym
from dqn_tensorflow import DQNAgent
from exploration import EpsilonGreedy, ThompsonSampling

LOAD_PRETRAINED = False
SAVE = True


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

for params in param_list:
    VERSION_NUM = params["version_num"]
    EPISODES_NUM = 100
    POLICY_DIR = f"./models/dqn/dqn-{VERSION_NUM}-e{EPISODES_NUM}-policy.keras"
    TARGET_DIR = f"./models/dqn/dqn-{VERSION_NUM}-e{EPISODES_NUM}-target.keras"
    PLOT_DIR = f"./models/dqn/tuning/{VERSION_NUM}.png"

    agent = DQNAgent(
        env = env, 
        input_size = 6, 
        output_size = 4, 
        learning_rate = params["learning_rate"],
        discount_factor = params["discount_factor"],
        exploration = EpsilonGreedy(
            epsilon = 1, 
            min_epsilon = 0.01, 
            epsilon_decay = params["epsilon_decay"]
        )
    )

    if LOAD_PRETRAINED:
        agent.load_models(POLICY_DIR, TARGET_DIR)
        agent.act()
    else:
        rewards = agent.train(EPISODES_NUM)
        print(rewards)

        with open('models/dqn/exploration_strats.txt','a') as f:
            f.write(f'[{",".join(str(x) for x in rewards)}]\n')
        
        utils.plot_rewards(rewards, PLOT_DIR)

        if SAVE:
            agent.policy_network.save(POLICY_DIR)
            agent.target_network.save(TARGET_DIR)