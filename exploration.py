from abc import ABC, abstractmethod
import numpy as np
import random

class ExplorationStrategy(ABC):
    @abstractmethod
    def get_action(self):
        pass

    def update(self):
        pass

"""
set epsilon to 1
if random number > epsilon, explore, else exploit
decay epsilon
    epsilon = epsilon * epsilon_decay
    epsilon cannot go below a minimum
"""
class EpsilonGreedy(ExplorationStrategy):
    def __init__(
            self,
            epsilon = 1,
            min_epsilon = 0.01,
            epsilon_decay = 0.995,
        ):

        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay


    def get_action(self):
        return(random.random() >= self.epsilon)
        
    def update(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)



"""
set all priors to flat, N ~ (0, 100) s.d. can be any arbitarily larger number
get posteriors, N ~ (mean, mu)
    mu = sqrt( (1 / mu_prior)^2 + num_of_visits)^(-1) )
    mean = (mu^2) * ( sum(visit_rewards) )
sample all distributions and pick action with highest sampled reward

""" 
class ThompsonSampling(ExplorationStrategy):
    def __init__(
        self,
        n_actions,
        alpha_prior = 1,
        beta_prior = 1
    ):
        self.n_actions = n_actions
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
        self.distributions = self._get_flat_priors()

    def _get_flat_priors(self):
        """
        Distribution list follows
        [mean, mu, n_times_actioned, total_reward_for_action]
        """
        return np.array([[self.alpha_prior, self.beta_prior] for _ in range(self.n_actions)])
    
    def _get_samples(self):
        return np.random.beta(self.distributions[:, 0], self.distributions[:, 1], self.n_actions)
    
    def get_action(self):
        return self._get_samples().argmax()
    
    def update(self, action, reward):
        # Update alpha and beta
        if (reward > 0):
            self.distributions[action, 0] += 1
        else:
            self.distributions[action, 1] += 1
        

"""
"""