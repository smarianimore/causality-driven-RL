from abc import ABC, abstractmethod


class AbstractAlgorithm(ABC):
    def __init__(self, env):
        self.env = env

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass

    @abstractmethod
    def save(self, path):
        pass

    @abstractmethod
    def load(self, path):
        pass

    @abstractmethod
    def set_hyperparameters(self, **kwargs):
        pass
