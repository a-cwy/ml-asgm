import environment
import gymnasium as gym
import numpy as np
from ppo import PPOAgent
from sac import SACAgent
from dqn import DQNAgent
from a2c import A2CAgent

environment.init()
env = gym.make('WaterHeater-v0')

# ppo = PPOAgent(
#     n_actions = 4, 
#     input_dims = (6,)
# )

# rewards_ppo = []

# for e in range(100):
#     print(f"Episode: {e + 1}")
#     total_ppo, bystep_ppo = ppo.act(env)
#     rewards_ppo.append(total_ppo)

# print(rewards_ppo)

# import res
# data = res.ppo_res_100().get()
# print(f"High: {max(data)}")
# print(f"Low: {min(data)}")
# print(f"Mean: {np.mean(data)}")
# print(f"S.D.: {np.std(data)}")



# sac = SACAgent()

# rewards_sac = []

# for e in range(100):
#     print(f"Episode: {e + 1}")
    
#     reward_total = 0.0
#     obs, _ = env.reset()
#     obs = sac.flatten_observation(obs)
#     done = False
    
#     while not done:
#         # deterministic=True for evaluation
#         action = sac.choose_action(obs, deterministic=True)
        
#         if isinstance(action, (list, np.ndarray)):
#             action = int(np.array(action).reshape(-1)[0])
#         else:
#             action = int(action)
            
#         obs_, reward, term, trunc, info = env.step(action)
#         done = term or trunc
#         obs_ = sac.flatten_observation(obs_)

#         obs = obs_
#         reward_total = reward_total + reward

#     rewards_sac.append(reward_total)

# print(rewards_sac)



# dqn = DQNAgent(
#     env,
#     6,
#     4
# )

# dqn.load_models('dash/target_dqn.keras')

# rewards_dqn = []

# for e in range(1):
#     print(f"Episode: {e + 1}")
#     rewards_dqn.append(np.sum(dqn.act()))

# print(rewards_dqn)



# a2c = A2CAgent(env)

# rewards_a2c = []

# for e in range(100):
#     print(f"Episode: {e + 1}")
#     rewards_a2c.append(np.sum(a2c.act()))

# print(rewards_a2c)