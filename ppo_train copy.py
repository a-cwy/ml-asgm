import numpy as np
from environment import WaterHeaterEnv
from ppo_agent_test import Agent  
import utils

LOAD_PRETRAINED = True

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
    # comfort_traj = []
    # hygiene_traj = []
    # energy_traj = []
    # safety_traj = []
    best_reward = -np.inf

    env = WaterHeaterEnv()
    agent = Agent(n_actions=env.action_space.n, 
                  input_dims=(6,), 
                  alpha=1e-3,
                  entropy_coef=0.01,
                  gamma=0.995,
                  gae_lambda=0.95,
                  policy_clip=0.2,
                  batch_size=512, 
                  n_epochs=4
                  )
    
    if(LOAD_PRETRAINED != True):
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

            # comfort_traj.append(info["rewards"]["comfort"])
            # hygiene_traj.append(info["rewards"]["hygiene"])
            # energy_traj.append(info["rewards"]["energy"])
            # safety_traj.append(info["rewards"]["safety"])


            if terminated or truncated:
                break
            if steps >= 2 * 7 * 96:  # safety cap (2 weeks)
                break

        # update policy after episode
        agent.learn()

        reward_history.append(score)

        if score > best_reward: 
            best_reward = score
            agent.actor.save_checkpoint()
            agent.critic.save_checkpoint()
            print(f"   Saved new best model checkpoints.(reward = {score:.2f})")

        print(f"Episode {ep+1}/{n_episodes} steps={steps} score={score:.3f}")
        print(f"   Comfort : {episode_comfort:.3f}")
        print(f"   Hygiene : {episode_hygiene:.3f}")
        print(f"   Energy  : {episode_energy:.3f}")
        print(f"   Safety  : {episode_safety:.3f}")
        print(f"   TOTAL   : {score:.3f}")
        # save models
        agent.actor.checkpoint_file = "models/ppo/actor_final.pth"
        agent.critic.checkpoint_file = "models/ppo/critic_final.pth"
        agent.save_models()
        print("Training finished.")

        # ===== Save Reward History =====

        np.save("ppo_rewards.npy",np.array(reward_history))
        print("Saved ppo_rewards.npy")

        # # ===== Training Summary Statistics =====
        # rewards = np.array(reward_history)

        # total_episodes = len(rewards)
        # final_reward = rewards[-1]
        # cumulative_sum = np.sum(rewards)
        # min_reward = np.min(rewards)
        # max_reward = np.max(rewards)
        # mean_reward = np.mean(rewards)
        # std_reward = np.std(rewards)
        # coef_variation = std_reward / abs(mean_reward) if mean_reward != 0 else 0.0
        # print("\n===== PPO Training Summary =====")
        # print(f"Episodes              : {total_episodes}")
        # print(f"Steps per Episode     : {steps}")
        # print(f"Final Episode Reward  : {final_reward:.3f}")
        # print(f"Cumulative Reward     : {cumulative_sum:.3f}")
        # print(f"Minimum Reward        : {min_reward:.3f}")
        # print(f"Maximum Reward        : {max_reward:.3f}")
        # print(f"Mean Reward           : {mean_reward:.3f}")
        # print(f"Std. Deviation        : {std_reward:.3f}")
        # print(f"Coeff. of Variation   : {coef_variation:.3f}")

        # np.save("comfort_cumsum.npy", np.cumsum(comfort_traj))
        # np.save("hygiene_cumsum.npy", np.cumsum(hygiene_traj))
        # np.save("energy_cumsum.npy", np.cumsum(energy_traj))
        # np.save("safety_cumsum.npy", np.cumsum(safety_traj))

        # print("Saved cumulative reward-by-type trajectories")

        # ===== Final Evaluation Run (NO TRAINING) =====
    else:
        print("\n===== Final Policy Evaluation =====")

        agent.actor.load_checkpoint()
        agent.critic.load_checkpoint()
        for ep in range(100):
            episode_reward, step_rewards = agent.act(env)
            reward_history.append(episode_reward)
            print(f"Episode {ep+1}/100  Reward: {episode_reward:.3f}")
        
        print("Rewards for 100 episodes: ", reward_history)

        utils.plot_rewards(reward_history,path="ppo_training_reward_plot.png")
        utils.plot_breakdown_cumulative(np.array(step_rewards))
        
        np.save("models/ppo/ppo_act_100_rewards.npy", reward_history)
        print("Saved step-level reward breakdown:")
        print("Shape:", step_rewards.shape)
        print("Shape:", np.array(reward_history).shape)
if __name__ == "__main__":  
    main()
