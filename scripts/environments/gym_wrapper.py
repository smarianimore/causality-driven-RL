import gymnasium as gym
from scripts.core.abstract_environment import AbstractEnvironment
from scripts.environments.wrappers import env_wrappers_manager


class GymWrapper(AbstractEnvironment):
    def __init__(self, env_name, config=None, seed: int = 42):
        self.env_name = env_name
        self.config = config or {}
        self.seed = seed
        self.env = self._create_env()

    def _create_env(self):
        valid_env_config = {k: v for k, v in self.config.items() if
                            k not in ['reward_wrapper', 'observation_wrapper', 'wrapper_kwargs', 'logging']}
        env = gym.make(self.env_name, **valid_env_config)
        env.action_space.seed(self.seed)
        env.observation_space.seed(self.seed)

        reward_wrapper_config = self.config.get('reward_wrapper')
        observation_wrapper_config = self.config.get('observation_wrapper')
        logging = self.config.get('logging', False)
        wrapper_kwargs = self.config.get('wrapper_kwargs', {})

        env = env_wrappers_manager(env, reward_wrapper_config, observation_wrapper_config, logging=logging,
                                   **wrapper_kwargs)
        return env

    def return_env(self):
        return self.env

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def render(self, mode='human'):
        self.env.render(mode)

    def close(self):
        self.env.close()

    def get_state_space(self):
        return self.env.observation_space

    def get_action_space(self):
        return self.env.action_space

