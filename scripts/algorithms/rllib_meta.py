from scripts.core.abstract_algorithm import AbstractAlgorithm
import ray
from ray.rllib.agents import ppo, dqn, a3c

class RLlibAlgorithm(AbstractAlgorithm):
    def __init__(self, env, algo_name, config):
        super().__init__(env)
        ray.init(ignore_reinit_error=True)
        self.algo_name = algo_name
        self.config = self._get_default_config(algo_name)
        self.config["env"] = env
        self.config.update(config)
        self.agent = self._initialize_agent(algo_name)

    def _get_default_config(self, algo_name):
        if algo_name == 'PPO':
            return ppo.DEFAULT_CONFIG.copy()
        elif algo_name == 'DQN':
            return dqn.DEFAULT_CONFIG.copy()
        elif algo_name == 'A3C':
            return a3c.DEFAULT_CONFIG.copy()
        else:
            raise ValueError(f"Unsupported RLlib algorithm: {algo_name}")

    def _initialize_agent(self, algo_name):
        if algo_name == 'PPO':
            return ppo.PPOTrainer(config=self.config, env=self.env)
        elif algo_name == 'DQN':
            return dqn.DQNTrainer(config=self.config, env=self.env)
        elif algo_name == 'A3C':
            return a3c.A3CTrainer(config=self.config, env=self.env)
        else:
            raise ValueError(f"Unsupported RLlib algorithm: {algo_name}")

    def train(self, episodes):
        for _ in range(episodes):
            self.agent.train()

    def evaluate(self, episodes):
        rewards = []
        for _ in range(episodes):
            reward = self.agent.evaluate()
            rewards.append(reward)
        return rewards

    def save(self, path):
        self.agent.save(path)

    def load(self, path):
        self.agent.restore(path)

    def set_hyperparameters(self, **kwargs):
        self.config.update(kwargs)
        self.agent = self._initialize_agent(self.algo_name)
