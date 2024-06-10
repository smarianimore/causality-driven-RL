import ray
from ray.rllib.algorithms import PPOConfig, DQNConfig


class RLlibAlgorithm:
    def __init__(self, environment, config):
        ray.init(ignore_reinit_error=True)
        self.env = environment
        self.algo_name = config['name'].lower()
        self.config = {
            "env": lambda _: environment,
            "num_workers": config['algo_config'].get('num_workers', 1),
            "num_gpus": config['algo_config'].get('num_gpus', 0),
        }
        self.trainer = self.get_algorithm_class()

    def get_algorithm_class(self):
        if self.algo_name == 'ppo':
            config = PPOConfig
        elif self.algo_name == 'dqn':
            config = DQNConfig
        else:
            raise ValueError(f"Unsupported algorithm: {self.algo_name}")

        config = config.environment(env=self.co)

        return algo

    def train(self, total_timesteps):
        iterations = total_timesteps // 1000  # assuming 1000 timesteps per iteration
        for i in range(iterations):
            result = self.trainer.train()
            if i % 10 == 0:  # print every 10 iterations
                print(f"Iteration {i}: reward_mean = {result['episode_reward_mean']}")

    def evaluate(self, num_episodes):
        total_reward = 0
        for i in range(num_episodes):
            done = False
            obs = self.env.reset()
            episode_reward = 0
            while not done:
                action = self.trainer.compute_action(obs)
                obs, reward, done, info = self.env.step(action)
                episode_reward += reward
            total_reward += episode_reward
            print(f"Episode {i + 1}: Reward = {episode_reward}")
        mean_reward = total_reward / num_episodes
        print(f"Mean reward over {num_episodes} episodes: {mean_reward}")
        return mean_reward

    def __del__(self):
        ray.shutdown()
