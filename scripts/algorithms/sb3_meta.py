from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3 import PPO, DQN, A2C
from scripts.algorithms.utils import StopTrainingOnEpisodesCallback, FinalPerformanceCallback
from scripts.core.abstract_algorithm import AbstractAlgorithm
import gymnasium as gym
import torch


class SB3Algorithm(AbstractAlgorithm):
    def __init__(self, env, algo_name, config, seed):
        super().__init__(env)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        config['device'] = device

        self.algo_name = algo_name
        self.seed = seed
        self.training_episodes = config.get('training_episodes', 1000)
        self.evaluation_episodes = config.get('evaluation_episodes', 100)
        self.model = self._initialize_model(algo_name, config)

    def _initialize_model(self, algo_name, config):
        policy = 'MlpPolicy' if isinstance(self.env.observation_space, gym.spaces.Discrete) else 'MultiInputPolicy'

        if self.algo_name == 'PPO':
            return PPO(policy, env=self.env, seed=self.seed, **config)
        elif self.algo_name == 'DQN':
            return DQN(policy, env=self.env, seed=self.seed, **config)
        elif self.algo_name == 'A2C':
            return A2C(policy, env=self.env, seed=self.seed, **config)
        else:
            raise ValueError(f"Unsupported SB3 algorithm: {algo_name}")

    def setup_callbacks(self):
        stop_training_callback = StopTrainingOnEpisodesCallback(self.training_episodes)
        performance_callback = FinalPerformanceCallback(self.algo_name)
        callbacks_list = [stop_training_callback, performance_callback]

        return CallbackList(callbacks_list), stop_training_callback, performance_callback

    def train(self):
        set_random_seed(self.seed)

        callback, stop_training_callback, performance_callback = self.setup_callbacks()
        self.model.learn(total_timesteps=int(1e50), callback=callback)

        total_time = stop_training_callback.get_total_time()
        episode_rewards = performance_callback.get_episode_rewards()
        episode_actions = performance_callback.get_episode_actions()

        return total_time, episode_rewards, episode_actions

    def evaluate(self):
        eval_mean_reward, eval_std_reward = evaluate_policy(self.model, self.env, n_eval_episodes=self.evaluation_episodes)

        self.env.close()

        res = f'{eval_mean_reward} \u00B1 {eval_std_reward}'
        return res

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        # TODO: to fix
        # self.model = self._initialize_model(self.algo_name, self.config)
        # self.model.load(path)
        pass

    def set_hyperparameters(self, **kwargs):
        self.model = self._initialize_model(self.algo_name, kwargs)
