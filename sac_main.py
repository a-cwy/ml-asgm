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
    # When constructing the agent, increase reward_scale so hygiene matters
    agent = Agent(
        input_dims=(flatten_observation(temp_obs).shape[0],),
        env=env,
        n_actions=4,
        alpha=0.0003,
        beta=0.0003,
        gamma=0.99,
        tau=0.005,
        batch_size=256,
        reward_scale=5.0,   # ↑ increase from 0.1 to 5.0 so reward signal is visible to critic
        max_size=100000,
        layer1_size=256,
        layer2_size=256,
        # ent_coef=2.0
    )
    
    n_games = 500

    
    filename = 'water_heater.png'
    figure_file = 'plots/' + filename

    if hasattr(env, 'reward_range'):
        best_score = env.reward_range[0]
    elif hasattr(env.unwrapped, 'reward_range'):
        best_score = env.unwrapped.reward_range[0]
    else:
        best_score = -np.inf
        
    score_history = []
    load_checkpoint = False

    if load_checkpoint:
        agent.load_models()
        env.render(mode='human')

    # (1 day = 96 steps) equals roughly 10 simulated days
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
        
        # store raw reward during warmup (optional: clip here if desired)
        agent.remember(observation, action, reward, observation_, done)
        
        if done:
            observation, _ = env.reset()
            observation = flatten_observation(observation)
        else:
            observation = observation_
    
    print(f"Warmup complete! Replay buffer size: {agent.memory.mem_cntr}")
    print("Starting training...\n")
    # ====================================================================

    # Inside training loop: track action histogram per episode
    for i in range(n_games):
        observation, info = env.reset()
        observation = flatten_observation(observation)
        done = False
        score = 0
        steps = 0

        # new: action counts
        action_counts = np.zeros(4, dtype=int)

        # reward breakdown per episode (comfort, hygiene, energy, safety)
        episode_reward_breakdown = [0.0, 0.0, 0.0, 0.0]

        while not done:
            # Use the agent's sampling policy for exploration during training.
            # If your Agent.choose_action supports `deterministic` you can pass it,
            # otherwise the default (stochastic) behavior should be used.
            try:
                action = agent.choose_action(observation, deterministic=False)
            except TypeError:
                # fallback if choose_action doesn't accept deterministic param
                action = agent.choose_action(observation)

            # ensure action is integer for discrete envs
            if isinstance(action, (list, np.ndarray)):
                # if returned as array, take first element
                action = int(np.array(action).reshape(-1)[0])
            else:
                action = int(action)

            action_counts[action] += 1

            observation_, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            observation_ = flatten_observation(observation_)

            # Accumulate reward breakdown if env provides it in info
            if isinstance(info, dict) and 'rewards' in info:
                r = info['rewards']
                episode_reward_breakdown[0] += float(r.get('comfort', 0.0))
                episode_reward_breakdown[1] += float(r.get('hygiene', 0.0))
                episode_reward_breakdown[2] += float(r.get('energy', 0.0))
                episode_reward_breakdown[3] += float(r.get('safety', 0.0))

            # Clip rewards to prevent extreme values used for learning
            clipped_reward = np.clip(reward, -10, 10)
            score += reward  # keep unclipped score for reporting
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

        # ====================================================================
        
        print(f'episode {i+1} / {n_games} score {score:.1f} avg_score {avg_score:.1f} steps {steps}')
        print("Action counts:", action_counts)   # see distribution of actions per episode
        if (i + 1) % 10 == 0:  # Show breakdown every 10 episodes
            print(utils.format_rewards(episode_reward_breakdown))
        # ====================================================================

    agent.save_models() # Force to save as 1st model for later loading without effecting previous best since change in code

    if not load_checkpoint:
        x = [i+1 for i in range(n_games)]
        utils.plot_rewards(score_history)