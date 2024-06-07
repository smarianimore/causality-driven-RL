from scripts.core.abstract_algorithm import AbstractAlgorithm
import ray
from ray.rllib.algorithms import PPOConfig, DQNConfig

# TODO
class RLlibAlgorithm(AbstractAlgorithm):
    def __init__(self, env, algo_name, config):
        super().__init__(env)
        ray.init(ignore_reinit_error=True)

        self.algo_name = algo_name
        self.training_episodes = config.get('training_episodes', 1000)
        self.evaluation_episodes = config.get('evaluation_episodes', 100)
        self.env = self._validate_env(env)
        self.agent = self._initialize_agent(algo_name)


    def _initialize_agent(self, algo_name):
        if algo_name == 'PPO':
            config_algo = PPOConfig()
        elif algo_name == 'DQN':
            config_algo = DQNConfig()
        else:
            raise ValueError(f"Unsupported RLlib algorithm: {algo_name}")

        algo = (config_algo
                .env_runners(num_env_runners=1)
                .resources(num_gpus=1)
                .framework("torch")
                .environment(env=self.env.env_name)
                .build()
                )

        return algo

    def _validate_env(self, env):
        # Perform environment validation for RLlib
        assert hasattr(env, 'reset'), "Environment must have a reset method"
        assert hasattr(env, 'step'), "Environment must have a step method"
        return env

    def train(self):
        for _ in range(self.training_episodes):
            self.agent.train()

    def evaluate(self):
        rewards = []
        for _ in range(self.evaluation_episodes):
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
