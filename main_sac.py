import gymnasium as gym
import numpy as np
from sac_torch import Agent
import utils

def flatten_observation(obs):
    """Convert gymnasium (Dict/Tuple) observation to 1D numpy array."""
    if isinstance(obs, dict):
        vec = np.array(list(obs.values()), dtype=np.float32).flatten()
    else:
        vec = np.array(obs, dtype=np.float32).flatten()
    return vec

utils.init()

if __name__ == '__main__':
    env = gym.make('WaterHeater-v0')
    
    temp_obs, _ = env.reset()
    
    # ====================================================================
    # STABILIZED HYPERPARAMETERS - Prevent catastrophic collapse
    # ====================================================================
    agent = Agent(
        input_dims=(flatten_observation(temp_obs).shape[0],),
        env=env,
        n_actions=4,
        alpha=0.0003,          # LOWER learning rate (was 0.001)
        beta=0.0003,           # LOWER learning rate (was 0.001)
        gamma=0.99,            
        tau=0.005,             
        batch_size=256,        # LARGER batch size (was 128) for stability
        reward_scale=0.1,      # MUCH LOWER reward scale (was 5)
        max_size=100000,       
        layer1_size=256,       # LARGER network (was 128)
        layer2_size=256        # LARGER network (was 128)
    )
    
    n_games = 100
    # ====================================================================
    
    filename = 'water_heater.png'
    figure_file = 'plots/' + filename

    if hasattr(env, 'reward_vector'):
        best_score = env.reward_vector[0]
    elif hasattr(env.unwrapped, 'reward_vector'):
        best_score = env.unwrapped.reward_vector[0]
    else:
        best_score = -np.inf
        
    score_history = []
    load_checkpoint = False

    if load_checkpoint:
        agent.load_models()
        env.render(mode='human')

    # ====================================================================
    # ADD: Warmup period - fill replay buffer before learning
    # ====================================================================
    print("Starting warmup period...")
    warmup_steps = 1000
    observation, _ = env.reset()
    observation = flatten_observation(observation)
    
    for step in range(warmup_steps):
        # Take random actions during warmup
        action = env.action_space.sample()
        observation_, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        observation_ = flatten_observation(observation_)
        
        agent.remember(observation, action, reward, observation_, done)
        
        if done:
            observation, _ = env.reset()
            observation = flatten_observation(observation)
        else:
            observation = observation_
    
    print(f"Warmup complete! Replay buffer size: {agent.memory.mem_cntr}")
    print("Starting training...\n")
    # ====================================================================

    for i in range(n_games):
        observation, info = env.reset()
        observation = flatten_observation(observation)
        done = False
        score = 0
        steps = 0
        
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            observation_ = flatten_observation(observation_)
            
            # ====================================================================
            # ADD: Clip rewards to prevent extreme values
            clipped_reward = np.clip(reward, -10, 10)
            # ====================================================================
            
            score += reward  # Track unclipped score
            agent.remember(observation, action, clipped_reward, observation_, done)
            
            if not load_checkpoint:
                agent.learn()
            
            observation = observation_
            steps += 1
        
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()

        print(f'episode {i+1} / {n_games} score {score:.1f} avg_score {avg_score:.1f} steps {steps}')

    # agent.save_models() # Used to save only best models

    if not load_checkpoint:
        x = [i+1 for i in range(n_games)]
        utils.plot_rewards(score_history)