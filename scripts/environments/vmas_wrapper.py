import random
import time
import pandas as pd
import torch
from tqdm.auto import tqdm
from vmas.simulator.environment import Wrapper
from vmas import make_env
from vmas.simulator.core import Agent
from vmas.simulator.utils import save_video
from scripts.core.env_interface import AbstractEnvironment
from stable_baselines3.common.vec_env import DummyVecEnv


class VMASWrapper(AbstractEnvironment):
    def __init__(self, scenario_name, config, algo_lib=None):
        self.config = config
        self.algo_lib = algo_lib
        self.algo = config.get('algorithm', 'PPO')
        self.render = config.get('render', False)
        self.save_render = config.get('save_render', False)
        self.num_envs = config.get('num_envs', 32)
        self.n_training_episodes = config.get('n_training_episodes', 100)
        self.random_action = config.get('random_action', False)
        self.device = config.get('device', 'cpu')
        self.scenario_name = scenario_name
        self.n_agents = config.get('n_agents', 4)
        self.continuous_actions = config.get('continuous_actions', True)
        self.visualize_render = config.get('visualize_render', True)
        # self.wrapper = config.get('wrapper', None)
        if self.algo_lib == 'sb3':
            self.wrapper = Wrapper.GYM
        elif self.algo_lib == 'vmas':
            self.wrapper = Wrapper.RLLIB
        else:
            self.wrapper = None
        self.logging = config.get('logging', False)
        self.env = None
        self.frame_list = []
        self.dfs = {}  # Dictionary to hold DataFrames for each environment
        self.columns_initialized = False  # Flag to indicate if columns are initialized
        self.initialize_env()

    def _get_deterministic_action(self, agent: Agent):
        if self.continuous_actions:
            action = -agent.action.u_range_tensor.expand(self.env.batch_dim, agent.action_size)
        else:
            action = (
                torch.tensor([1], device=self.env.device, dtype=torch.long)
                .unsqueeze(-1)
                .expand(self.env.batch_dim, 1)
            )
        return action.clone()

    def initialize_env(self):
        dict_spaces = True
        self.env = make_env(
            scenario=self.scenario_name,
            num_envs=self.num_envs,
            device=self.device,
            continuous_actions=self.continuous_actions,
            dict_spaces=dict_spaces,
            wrapper=self.wrapper,
            seed=42,
            n_agents=self.n_agents,
        )

        """if self.algo_lib == 'sb3':
            self.env = DummyVecEnv([lambda: self.env])"""

    def initialize_dataframes(self, observations):
        # Initialize the DataFrames based on the first observation
        columns = []
        for agent_id in range(self.n_agents):
            num_sensors = observations[f'agent_{agent_id}'].shape[1] - 6  # Subtracting 6 for PX, PY, VX, VY, DX, DY
            agent_columns = [f"agent_{agent_id}_{feature}" for feature in
                             ['PX', 'PY', 'VX', 'VY', 'DX', 'DY'] + [f'sensor{N}' for N in range(num_sensors)]]
            agent_columns.append(f"reward_agent_{agent_id}")
            agent_columns.append(f"action_agent_{agent_id}")
            columns.extend(agent_columns)

        for env_id in range(self.num_envs):
            self.dfs[env_id] = pd.DataFrame(columns=columns)
        self.columns_initialized = True

    def update_dataframe(self, observations, rewards, actions, env_id):
        data = []
        for agent_id in range(self.n_agents):
            agent_obs = observations[f'agent_{agent_id}'][env_id].cpu().numpy()
            agent_reward = rewards[f'agent_{agent_id}'][env_id].cpu().item()

            if isinstance(actions, dict):
                agent_action = actions[f'agent_{agent_id}'][env_id].cpu().numpy() if self.continuous_actions else \
                    actions[f'agent_{agent_id}'][env_id].cpu().item()
            else:
                agent_action = actions[agent_id][env_id].cpu().numpy() if self.continuous_actions else \
                    actions[agent_id][env_id].cpu().item()

            data.extend(agent_obs)
            data.append(agent_reward)
            data.append(agent_action)
        self.dfs[env_id].loc[len(self.dfs[env_id])] = data

    def reset(self):
        return self.env.reset()

    def step(self, action):
        obs, rews, dones, info = self.env.step(action)
        if self.logging:
            if not self.columns_initialized:
                self.initialize_dataframes(obs)
            for env_id in range(self.num_envs):
                self.update_dataframe(obs, rews, action, env_id)
        return obs, rews, dones, info

    def render(self, mode='human'):
        frame = self.env.render(
            mode="rgb_array",
            agent_index_focus=None,
            visualize_when_rgb=self.visualize_render,
        )
        if self.save_render:
            self.frame_list.append(frame)
        return frame

    def close(self):
        self.env.close()

    def get_logs(self):
        return self.dfs

    def return_env(self):
        return self.env

    def get_state_space(self):
        return self.env.observation_space

    def get_action_space(self):
        return self.env.action_space
