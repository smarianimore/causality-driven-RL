# sb3_algorithm.py
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.evaluation import evaluate_policy


class SB3Algorithm:
    def __init__(self, environment, config):
        self.env = environment
        algo_name = config['name']
        self.model = globals()[algo_name]('MultiInputPolicy', self.env, **config.get('algo_config', {}))

    def train(self, total_timesteps):
        self.model.learn(total_timesteps=total_timesteps)

    def evaluate(self, num_episodes):
        mean_reward, std_reward = evaluate_policy(self.model, self.env, n_eval_episodes=num_episodes)
        print(f"Mean reward: {mean_reward} +/- {std_reward}")
        return mean_reward, std_reward

