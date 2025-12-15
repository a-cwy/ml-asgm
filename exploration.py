from abc import ABC, abstractmethod
import random

class ExplorationStrategy(ABC):
    @abstractmethod
    def eval(self):
        pass

    @abstractmethod
    def decay(self):
        pass

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

    def decay(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)

    def eval(self):
        return(random.random() >= self.epsilon)