import os
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import numpy as np

class ActorNetwork(nn.Module):
    def __init__(
            self, 
            n_actions, 
            input_dims, 
            alpha,
            fc1_dims=256, 
            fc2_dims=256, 
            chkpt_dir='dash/actor_ppo.pth'
        ):

        super(ActorNetwork, self).__init__()

        # os.makedirs(chkpt_dir,exist_ok=True)
        self.checkpoint_file = chkpt_dir

        self.actor = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, n_actions),
            nn.Softmax(dim=-1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        probs = self.actor(state)
        dist = Categorical(probs)
        return dist

    def load_checkpoint(self):
        print(self.checkpoint_file)
        self.load_state_dict(T.load(self.checkpoint_file, map_location = T.device('cpu')))

class PPOAgent:
    def __init__(
        self,
        n_actions,
        input_dims,
        alpha=1e-4
    ):

        self.actor = ActorNetwork(n_actions, input_dims, alpha)
        self.actor.load_checkpoint()


    def _process_obs(self, obs):
        return np.array([
            obs["day"],
            obs["time"],
            obs["waterTemperature"],
            obs["targetTemperature"],
            obs["timeSinceSterilization"],
            obs["forecast"]
        ], dtype=np.float32)
    
    def act(self, env):
        """
        Evaluate the trained PPO policy (NO learning).
        Runs one deterministic episode.
        """
        obs, _ = env.reset()
        state = self._process_obs(obs)

        terminated = False
        truncated = False
        steps = 0
        total_reward = 0.0

        step_rewards = []

        while not terminated and not truncated:
            state_t = T.tensor(state, dtype=T.float32).unsqueeze(0).to(self.actor.device)

        # Deterministic action (no exploration)
            with T.no_grad():
                dist = self.actor(state_t)
                action = dist.probs.argmax(dim=-1).item()

            next_obs, reward, terminated, truncated, info = env.step(action)

            comfort = info["rewards"]["comfort"]
            hygiene = info["rewards"]["hygiene"]
            energy = info["rewards"]["energy"]
            safety = info["rewards"]["safety"]
        
            step_rewards.append([comfort, hygiene, energy, safety])

            total_reward += reward
            state = self._process_obs(next_obs)
            steps += 1

        step_rewards = np.array(step_rewards)
        return total_reward, step_rewards

