from scripts.core.abstract_algorithm import AbstractAlgorithm
from stable_baselines3 import PPO, DQN, A2C

class SB3Algorithm(AbstractAlgorithm):
    def __init__(self, env, algo_name, config):
        super().__init__(env)
        self.algo_name = algo_name
        self.model = self._initialize_model(algo_name, config)

    def _initialize_model(self, algo_name, config):
        if algo_name == 'PPO':
            return PPO('MlpPolicy', self.env, **config)
        elif algo_name == 'DQN':
            return DQN('MlpPolicy', self.env, **config)
        elif algo_name == 'A2C':
            return A2C('MlpPolicy', self.env, **config)
        else:
            raise ValueError(f"Unsupported SB3 algorithm: {algo_name}")

    def train(self, episodes):
        self.model.learn(total_timesteps=episodes)

    def evaluate(self, episodes):
        rewards = []
        for _ in range(episodes):
            obs = self.env.reset()
            total_reward = 0
            done = False
            while not done:
                action, _ = self.model.predict(obs)
                obs, reward, done, info = self.env.step(action)
                total_reward += reward
            rewards.append(total_reward)
        return rewards

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = self._initialize_model(self.algo_name, self.env)
        self.model.load(path)

    def set_hyperparameters(self, **kwargs):
        self.model = self._initialize_model(self.algo_name, kwargs)
