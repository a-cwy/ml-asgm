# train.py
import numpy as np
from environment import WaterHeaterEnv
from ppo_agent import Agent  

def flatten_obs_dict(obs_dict):
    """Convert env observation dict to flat vector of 6 features."""
    return np.array([
        float(obs_dict["day"]),
        float(obs_dict["time"]),
        float(obs_dict["waterTemperature"].reshape(-1)[0]),
        float(obs_dict["targetTemperature"].reshape(-1)[0]),
        float(obs_dict["timeSinceSterilization"]),
        float(obs_dict["forecast"])
    ], dtype=np.float32)

def main():
    reward_history = []
    env = WaterHeaterEnv()
    agent = Agent(n_actions=env.action_space.n, input_dims=(6,), batch_size=512, n_epochs=4)

    n_episodes = 300  # small number to verify output quickly; increase for real training
    for ep in range(n_episodes):
        obs, info = env.reset()
        state = flatten_obs_dict(obs)

        done = False
        score = 0.0
        steps = 0

        episode_comfort = 0.0
        episode_hygiene = 0.0
        episode_energy = 0.0
        episode_safety = 0.0

        # run one episode (we'll cap to a week as safety)
        while True:
            action, logprob, val = agent.choose_action(state)
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_state = flatten_obs_dict(next_obs)

            #scaled_reward = reward * 0.01  # scale reward to keep values, target is beteen -20000-1000
            agent.remember(state, action, logprob, val, reward, float(terminated or truncated))
            score += reward

            state = next_state
            steps += 1

            episode_comfort += info["rewards"]["comfort"]
            episode_hygiene += info["rewards"]["hygiene"]
            episode_energy += info["rewards"]["energy"]
            episode_safety += info["rewards"]["safety"]

            if terminated or truncated:
                break
            if steps >= 2 * 7 * 96:  # safety cap (2 weeks)
                break

        # update policy after episode
        agent.learn()

        reward_history.append(score)

        print(f"Episode {ep+1}/{n_episodes} steps={steps} score={score:.3f}")
        print(f"   Comfort : {episode_comfort:.3f}")
        print(f"   Hygiene : {episode_hygiene:.3f}")
        print(f"   Energy  : {episode_energy:.3f}")
        print(f"   Safety  : {episode_safety:.3f}")
        print(f"   TOTAL   : {score:.3f}")
    # save models
    agent.save_models()
    print("Training finished.")

    np.save("ppo_rewards.npy",np.array(reward_history))
    print("Saved ppo_rewards.npy")

if __name__ == "__main__":  
    main()

