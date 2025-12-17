import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from sac_torch import Agent
import utils
from sac_main import flatten_observation

# Initialize utilities
utils.init()

def train_single_agent(env, agent_config, n_games=100, label="Agent"):
    """
    Trains a single SAC agent and returns its score history and the trained agent.
    """
    agent = Agent(**agent_config)
    score_history = []
    best_score = -np.inf
    
    print(f"\n--- Starting Training: {label} ---")
    
    # Warmup
    observation, _ = env.reset()
    observation = flatten_observation(observation)
    for _ in range(1000): 
        action = env.action_space.sample()
        obs_, reward, term, trunc, _ = env.step(action)
        done = term or trunc
        obs_ = flatten_observation(obs_)
        agent.remember(observation, action, reward, obs_, done)
        if done:
            observation, _ = env.reset()
            observation = flatten_observation(observation)
        else:
            observation = obs_

    # Main Training Loop
    for i in range(n_games):
        observation, _ = env.reset()
        observation = flatten_observation(observation)
        done = False
        score = 0
        
        while not done:
            # Note: deterministic=False during training for exploration
            action = agent.choose_action(observation)
            
            if isinstance(action, (list, np.ndarray)):
                action = int(np.array(action).reshape(-1)[0])
            else:
                action = int(action)

            observation_, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            observation_ = flatten_observation(observation_)
            
            clipped_reward = np.clip(reward, -10, 10)
            score += reward
            
            agent.remember(observation, action, clipped_reward, observation_, done)
            agent.learn()
            
            observation = observation_

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        
        if avg_score > best_score:
            best_score = avg_score


        print(f'{label} | Episode {i+1}/{n_games} | Score: {score:.1f} | Avg: {avg_score:.1f}')

    return agent, score_history

def evaluate_episode_breakdown(env, agent):
    """
    Runs one episode and returns the cumulative history for each reward component.
    """
    obs, _ = env.reset()
    obs = flatten_observation(obs)
    done = False
    
    # Initialize lists to store cumulative sums
    keys = ['comfort', 'hygiene', 'energy', 'safety']
    history = {k: [0.0] for k in keys} # start at 0
    current_sums = {k: 0.0 for k in keys}
    
    while not done:
        # deterministic=True for evaluation
        action = agent.choose_action(obs, deterministic=True)
        
        if isinstance(action, (list, np.ndarray)):
            action = int(np.array(action).reshape(-1)[0])
        else:
            action = int(action)
            
        obs_, reward, term, trunc, info = env.step(action)
        done = term or trunc
        obs_ = flatten_observation(obs_)
        
        # Extract breakdown from info['rewards'] if available
        # Default to 0.0 if key is missing
        if isinstance(info, dict) and 'rewards' in info:
            r_dict = info['rewards']
            for k in keys:
                val = float(r_dict.get(k, 0.0))
                current_sums[k] += val
                history[k].append(current_sums[k])
        else:
            # Fallback if env doesn't provide breakdown
            for k in keys:
                history[k].append(history[k][-1])

        obs = obs_
        
    return history

if __name__ == '__main__':
    env = gym.make('WaterHeater-v0')
    temp_obs, _ = env.reset()
    input_dim = flatten_observation(temp_obs).shape[0]

    # --- Config ---
    base_config = {
        'input_dims': (input_dim,), 'env': env, 'n_actions': 4,
        'gamma': 0.99, 'tau': 0.005, 'batch_size': 256,
        'reward_scale': 1.0, 
        'ent_coef': 0.5,
        'alpha': 0.001, 'beta': 0.001,
    }

    # Define differences
    config_baseline = base_config.copy()
    config_baseline.update({'alpha': 0.0003, 'beta': 0.0003})

    config_high_lr = base_config.copy()
    config_high_lr.update({'alpha': 0.003, 'beta': 0.003})
    
    config_low_lr = base_config.copy()
    config_low_lr.update({'alpha': 0.00003, 'beta': 0.00003})

    # --- Train ---
    
    agent_baseline, scores_base = train_single_agent(env, config_baseline, label="Baseline")
    agent_high, scores_high = train_single_agent(env, config_high_lr, label="High LR")
    agent_low, scores_low = train_single_agent(env, config_low_lr, label="Low LR")

    # --- Plot 1: Total Scores Comparison ---
    plt.figure(figsize=(10, 5))
    plt.plot(scores_base, label='Baseline (3e-4)', alpha=0.6)
    plt.plot(scores_high, label='High LR (3e-3)', alpha=0.6)
    plt.plot(scores_low, label='Low LR (3e-5)', alpha=0.6)
    plt.title('Training Comparison: Total Reward')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/training_comparison.png')
    plt.show()

    # --- Plot 2: Cumulative Component Breakdown (Subplots) ---
    print("\nEvaluating for Component Breakdown...")
    data_baseline = evaluate_episode_breakdown(env, agent_baseline)
    data_high_lr = evaluate_episode_breakdown(env, agent_high)

    # Create 2 Subplots vertically
    fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    # Helper to plot one dictionary onto an axis
    def plot_components(ax, data, title):
        steps = range(len(data['comfort']))
        ax.plot(steps, data['comfort'], label='Comfort', color='green')
        ax.plot(steps, data['hygiene'], label='Hygiene', color='blue')
        ax.plot(steps, data['energy'], label='Energy', color='red')
        ax.plot(steps, data['safety'], label='Safety', color='orange')
        ax.set_title(title)
        ax.set_ylabel('Cumulative Reward')
        ax.grid(True)
        ax.legend()

    # Plot Baseline
    plot_components(axes[0], data_baseline, "Baseline Agent: Cumulative Reward Breakdown")
    
    # Plot High LR
    plot_components(axes[1], data_high_lr, "High LR Agent: Cumulative Reward Breakdown")

    plt.xlabel('Steps (Single Episode)')
    plt.tight_layout()
    plt.savefig('plots/component_breakdown.png')
    plt.show()

    print("All plots generated and saved.")