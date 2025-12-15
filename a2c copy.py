## FOR BACKUP PURPOSES ONLY

import sys
import gymnasium as gym
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import os
import utils


# Actor network (PyTorch)
class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_size), 
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.net(x)


class Critic(nn.Module):
    def __init__(self, state_size, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        return self.net(x)


class A2CAgent():
    """
    Simple A2C agent using PyTorch. Supports n-step returns.
    If `actor_model` or `critic_model` are provided they should be nn.Module instances;
    otherwise defaults are created using observation shape inferred from the env.
    """
    def __init__(
            self,
            env: gym.Env,
            actor_model: nn.Module = None,
            critic_model: nn.Module = None,
            discount_factor: float = 0.99,
            learning_rate: float = 1e-4,  #test with 1e-4, 3e-4, 5e-4
            gamma: float = 0.99,
            n_steps: int = 50,
            entropy_coef: float = 0.01,
            value_loss_coef: float = 0.5,
            max_grad_norm: float = 0.5
        ):
        self.env = env
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.n_steps = n_steps
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm

        # Build a flattened state vector from a sample observation
        sample_obs, _ = self.env.reset()
        sample_vec = self._flatten_obs(sample_obs)
        self.state_size = sample_vec.shape[0]
        self.action_size = self.env.action_space.n

        # Create models if not provided
        self.actor_model = actor_model or Actor(self.state_size, self.action_size)
        self.critic_model = critic_model or Critic(self.state_size)

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.actor_model.to(self.device)
        self.critic_model.to(self.device)

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor_model.parameters(), lr=self.learning_rate)
        self.critic_optimizer = optim.Adam(self.critic_model.parameters(), lr=self.learning_rate)

    def _flatten_obs(self, obs):
        """Convert gym `Dict` observation to 1D numpy float32 array (same as DQN)."""
        if isinstance(obs, dict):
            vec = np.array(list(obs.values()), dtype=np.float32).flatten()
        else:
            vec = np.array(obs, dtype=np.float32).flatten()
        return vec

    def choose_action(self, state_vec):
        """Sample an action from the actor given a flattened state vector."""
        state = torch.tensor(state_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            probs = self.actor_model(state)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), dist.entropy()

    def act(self):
        """Run one episode using the current policy (deterministic argmax)."""
        #breakdown rewards
        obs, _ = self.env.reset()
        state = self._flatten_obs(obs)
        terminated = False
        truncated = False
        total_reward = 0.0
        while (not terminated and not truncated):
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                probs = self.actor_model(state_t)
                action = probs.argmax(dim=-1).item()

            next_obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            state = self._flatten_obs(next_obs)
            rewards_breakdown = np.add(np.zeros(len(info["rewards"])), list(info["rewards"].values()))
            # if terminated or truncated:
            #     break
        print(f"ACT: Episode rewards breakdown {utils.format_rewards(rewards_breakdown)}")

    def save_models(self, actor_path, critic_path):
        """
        Save actor and critic state_dicts to the given paths. Also saves an
        optional combined checkpoint including optimizer states and epoch when
        called with `save_checkpoint=True` (uses the agent's optimizers by
        default).
        """
        os.makedirs(os.path.dirname(actor_path) or '.', exist_ok=True)
        os.makedirs(os.path.dirname(critic_path) or '.', exist_ok=True)
        torch.save(self.actor_model.state_dict(), actor_path)
        torch.save(self.critic_model.state_dict(), critic_path)
        print(f"Saved actor model to {actor_path}")
        print(f"Saved critic model to {critic_path}")

    def save_checkpoint(self, actor_path, critic_path, checkpoint_path=None, epoch=None, optimizer_actor=None, optimizer_critic=None):
        """
        Save model weights and (optionally) optimizer states into a single
        checkpoint file. If `optimizer_actor` / `optimizer_critic` are not
        provided, the agent's own optimizers are used when available.
        """
        # ensure model weights are saved as well
        self.save_models(actor_path, critic_path)

        if checkpoint_path is None:
            checkpoint_path = os.path.join(os.path.dirname(actor_path) or '.', 'a2c_checkpoint.pth')

        os.makedirs(os.path.dirname(checkpoint_path) or '.', exist_ok=True)

        if optimizer_actor is None and hasattr(self, 'actor_optimizer'):
            optimizer_actor = self.actor_optimizer
        if optimizer_critic is None and hasattr(self, 'critic_optimizer'):
            optimizer_critic = self.critic_optimizer

        ckpt = {
            'epoch': epoch,
            'actor_state_dict': self.actor_model.state_dict(),
            'critic_state_dict': self.critic_model.state_dict(),
        }

        if optimizer_actor is not None:
            ckpt['optimizer_actor'] = optimizer_actor.state_dict()
        if optimizer_critic is not None:
            ckpt['optimizer_critic'] = optimizer_critic.state_dict()

        torch.save(ckpt, checkpoint_path)
        print(f"Saved full checkpoint to {checkpoint_path}")

    def load_models(self, actor_path, critic_path, map_location=None, checkpoint_path=None, load_optimizers: bool = False):
        """
        Load actor and critic weights from the provided paths. If `load_optimizers`
        is True and a `checkpoint_path` exists, the optimizer states will be
        restored into the agent's optimizers when available. Returns the epoch
        value stored in the checkpoint (or None).
        """
        if map_location is None:
            map_location = self.device

        ## Load actor
        if os.path.exists(actor_path):
            self.actor_model.load_state_dict(torch.load(actor_path, map_location=map_location))
            print(f"Loaded actor model from {actor_path}")
        else:
            print(f"Warning: actor model path not found: {actor_path}")
        
        ## Load critic
        if os.path.exists(critic_path):
            self.critic_model.load_state_dict(torch.load(critic_path, map_location=map_location))
            print(f"Loaded critic model from {critic_path}")
        else:
            print(f"Warning: critic model path not found: {critic_path}")

        loaded_epoch = None
        if load_optimizers:
            if checkpoint_path is None:
                checkpoint_path = os.path.join(os.path.dirname(actor_path) or '.', 'a2c_checkpoint.pth')

            if os.path.exists(checkpoint_path):
                ckpt = torch.load(checkpoint_path, map_location=map_location)
                # restore optimizers if available on agent
                if hasattr(self, 'actor_optimizer') and 'optimizer_actor' in ckpt:
                    try:
                        self.actor_optimizer.load_state_dict(ckpt['optimizer_actor'])
                    except Exception as e:
                        print(f"Warning: failed to load actor optimizer state: {e}")

                if hasattr(self, 'critic_optimizer') and 'optimizer_critic' in ckpt:
                    try:
                        self.critic_optimizer.load_state_dict(ckpt['optimizer_critic'])
                    except Exception as e:
                        print(f"Warning: failed to load critic optimizer state: {e}")

                # refresh model weights from checkpoint if present
                if 'actor_state_dict' in ckpt:
                    self.actor_model.load_state_dict(ckpt['actor_state_dict'])
                if 'critic_state_dict' in ckpt:
                    self.critic_model.load_state_dict(ckpt['critic_state_dict'])

                loaded_epoch = ckpt.get('epoch', None)
                print(f"Loaded checkpoint from {checkpoint_path} (epoch={loaded_epoch})")

        return loaded_epoch

    def compute_n_step_returns(self, rewards, next_value, masks):
        """Compute discounted n-step returns (list) in reverse order."""
        R = next_value
        returns = []
        for r, m in zip(reversed(rewards), reversed(masks)):
            R = r + self.gamma * R * m
            returns.insert(0, R)
        return returns

    def train(self, episodes: int, save_every: int = 500, actor_path=None, critic_path=None, max_steps_per_episode: int = 2000):
        """
        Train the actor-critic using n-step returns.
        """
        rewards_per_episode = []

        #for each episode
        for ep in range(episodes):
            rewards_breakdown = [0, 0, 0, 0]
            obs, _ = self.env.reset()
            state = self._flatten_obs(obs)
            ep_reward = 0.0

            log_probs = []
            values = []
            rewards = []
            entropies = []
            masks = []  # 1.0 for not done, 0.0 for done

            #for each step in episode
            for step in range(max_steps_per_episode):
                state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

                probs = self.actor_model(state_t)
                dist = Categorical(probs)
                action = dist.sample()

                value = self.critic_model(state_t)
                log_prob = dist.log_prob(action)
                entropy = dist.entropy()

                # use info 
                next_obs, reward, terminated, truncated, info = self.env.step(action.item())
                done = bool(terminated or truncated)

                log_probs.append(log_prob)
                values.append(value)
                rewards.append(float(reward))
                entropies.append(entropy)
                masks.append(0.0 if done else 1.0)

                ep_reward += reward
                state = self._flatten_obs(next_obs)
                rewards_breakdown = np.add(rewards_breakdown, list(info["rewards"].values()))

                # if we have enough steps or episode finished, perform update
                if len(rewards) >= self.n_steps or done:
                    # bootstrap value
                    if done:
                        next_value = 0.0
                    else:
                        next_state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                        with torch.no_grad():
                            next_value = self.critic_model(next_state_t).item()

                    returns = self.compute_n_step_returns(rewards, next_value, masks)

                    returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
                    values_t = torch.cat(values).squeeze(-1)
                    log_probs_t = torch.cat(log_probs)
                    entropies_t = torch.cat(entropies)

                    advantages = returns_t - values_t

                    actor_loss = -(log_probs_t * advantages.detach()).mean() - self.entropy_coef * entropies_t.mean()
                    critic_loss = F.mse_loss(values_t, returns_t)

                    # optimize actor
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.actor_model.parameters(), self.max_grad_norm)
                    self.actor_optimizer.step()

                    # optimize critic
                    self.critic_optimizer.zero_grad()
                    (self.value_loss_coef * critic_loss).backward()
                    torch.nn.utils.clip_grad_norm_(self.critic_model.parameters(), self.max_grad_norm)
                    self.critic_optimizer.step()

                    # clear buffers
                    log_probs = []
                    values = []
                    rewards = []
                    entropies = []
                    masks = []

                if done:
                    break

            rewards_per_episode.append(ep_reward)

            
            # print breakdown
            print(f"EPISODE {ep} / {episodes}")
            print(f"{utils.format_rewards(rewards_breakdown)}")
            if ep % 10 == 0:
                avg_last_10 = np.mean(rewards_per_episode[-10:])
                print(f"  Avg reward (last 10): {avg_last_10:.2f}")
            
            # save checkpoint
            if actor_path and critic_path and (ep % save_every == 0) and (ep > 0):
                self.save_models(actor_path, critic_path)

        return rewards_per_episode


#### TRAINING CONFIG
from datetime import datetime
LOAD_PRETRAINED = False
USE_DATESTAMP = False

# Get the current date and time
now = datetime.now()

# Format the datetime object into a string
# YYYYMMDD_HHMMSS is a common and sortable format
version = now.strftime("%Y%m%d_%H%M%S") if USE_DATESTAMP else "v4"  #CHANGE THIS TO DESIRED VERSION STRING

VERSION_NUM = f"{version}"
episodes = 500  #CHANGE THIS TO DESIRED NUMBER OF EPISODES

# Define model save paths
# ACTOR_DIR = f"./models/a2c/a2c-20251204_234342-e200-actor.pth"
# CRITIC_DIR = f"./models/a2c/a2c-20251204_234342-e200-critic.pth"

ACTOR_DIR = f"./models/a2c/a2c-{VERSION_NUM}-e{episodes}-actor.pth"
CRITIC_DIR = f"./models/a2c/a2c-{VERSION_NUM}-e{episodes}-critic.pth"

if __name__ == "__main__":
    utils.init()
    env = gym.make("WaterHeater-v0")
    agent = A2CAgent(env)
    if len(sys.argv) > 1:
        try:
            int(sys.argv[1])
            episodes = int(sys.argv[1])
            print(f"Training for {episodes} episodes.")
        except ValueError:
            print(f"Invalid number of episodes provided. Using default ({episodes}).")
    
    #ENSURE directory EXISTS
    if LOAD_PRETRAINED and os.path.exists(ACTOR_DIR) and os.path.exists(CRITIC_DIR):
        agent.load_models(ACTOR_DIR, CRITIC_DIR)
        agent.act()
    else:
        rewards = agent.train(episodes, actor_path=ACTOR_DIR, critic_path=CRITIC_DIR)
        print(f"Rewards for {episodes} episodes: {rewards}")
        #print(type(rewards)) #debug
        np.save(f"./models/a2c/a2c-{VERSION_NUM}-e{episodes}-rewards.npy", np.array(rewards))
        if not os.path.exists(os.path.dirname(ACTOR_DIR)):
            os.makedirs(os.path.dirname(ACTOR_DIR), exist_ok=True)
        
        agent.save_models(ACTOR_DIR, CRITIC_DIR)
        #Plot rewards
        utils.plot_rewards(rewards)
