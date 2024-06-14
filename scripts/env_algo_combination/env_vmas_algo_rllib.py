import os
import ray
from gymnasium.spaces import Box, Discrete
from ray.tune.registry import register_env
import numpy as np
from vmas import make_env, Wrapper
from typing import Dict, Optional


class EnvVmasAlgoRLlib:
    def __init__(self, env_config, algo_config, sim_config):
        ray.init(ignore_reinit_error=True)
        self.env_config = env_config
        self.algo_config = algo_config
        self.sim_config = sim_config

        self.register_environment()

        self.algo_config_rllib = self.create_algorithm()

        self.set_simulation_parameters()

    def register_environment(self):
        register_env(self.env_config['scenario_name'], lambda config: self.env_creator())

    def env_creator(self):
        config = self.env_config['config_env']

        env = make_env(
            scenario=self.env_config["scenario_name"],
            num_envs=config["num_envs"],
            device=config["device"],
            continuous_actions=config["continuous_actions"],
            wrapper=Wrapper.RLLIB,
            max_steps=config["max_steps"],
            n_agents=config["n_agents"],
        )
        env.observation_space = Box(low=0, high=1, shape=(config["observation_shape"],), dtype=np.float32)
        env.action_space = Discrete(config["action_space"])
        return env

    """    def create_algorithm(self):
        self.algo_name = self.algo_config['name'].upper()
        algo_config_class = {
            "PPO": PPOConfig,
            "DQN": DQNConfig
        }.get(self.algo_name)
        if not algo_config_class:
            raise ValueError(f"Unsupported algorithm: {self.algo_name}")

        algo_config_instance = algo_config_class().to_dict()
        return algo_config_instance"""

    def set_simulation_parameters(self):
        self.training_episodes = self.sim_config['training_episodes']
        self.evaluation_episodes = self.sim_config['evaluation_episodes']
        self.seed = self.sim_config['seed']


