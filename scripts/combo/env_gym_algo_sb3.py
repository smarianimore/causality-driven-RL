import gymnasium as gym
from stable_baselines3.common.evaluation import evaluate_policy
from scripts.utils import dynamic_import
from stable_baselines3.ppo import PPO
from stable_baselines3.dqn import DQN


class EnvGymAlgoSB3:
    def __init__(self, env_config, algo_config):
        self.env = self.create_environment(env_config)
        self.model = self.create_algorithm(algo_config)

    def create_environment(self, config):
        env = gym.make(config['name'], **config.get('config_env', {}))
        for wrapper in config.get('wrappers', {}).values():
            module, class_name = wrapper['wrapper'].rsplit('.', 1)
            WrapperClass = dynamic_import(module, class_name)
            env = WrapperClass(env, **wrapper.get('kwargs', {}))
        return env

    def create_algorithm(self, config):
        algo_name = config['name']
        return globals()[algo_name.upper()]('MultiInputPolicy', self.env, **config.get('algo_config', {}))

    def train(self, training_episode, seed):
        self.model.learn(total_timesteps=1e50)

    def evaluate(self, evaluation_episode, seed):
        mean_reward, std_reward = evaluate_policy(self.model, self.env, n_eval_episodes=evaluation_episode)
        print(f"Mean reward: {mean_reward} +/- {std_reward}")
        return mean_reward, std_reward
