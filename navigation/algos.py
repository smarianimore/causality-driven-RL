from typing import Dict
import random
import numpy as np
import pandas as pd
import torch
from torch import tensor
from vmas.simulator.environment import Environment
from navigation.causality_algos import CausalInferenceForRL, CausalDiscovery
from navigation.utils import detach_dict
from path_repo import GLOBAL_PATH_REPO


class RandomAgentVMAS:
    def __init__(self, env, device, seed: int = 42):
        self.env = env
        self.device = device

        np.random.seed(seed)
        random.seed(42)

    def choose_action(self, obs: Dict = None):
        action = torch.randint(
            low=0,
            high=9,
            size=(1,),
            device=self.device,
        )
        return action

    def update(self, obs: Dict = None, action: float = None, reward: float = None, next_obs: Dict = None):
        pass


class CausalAgentVMAS:
    def __init__(self, env: Environment, device, agent_id: int = 0, timesteps_for_causality_update: int = 10000, seed: int = 42):

        self.env = env
        self.action_space_size = self.env.action_space['agent_0'].n
        self.device = device
        self.agent_id = agent_id
        self.scenario = 'navigation' + f'_agent_{self.agent_id}'
        self.timesteps_for_causality_update = timesteps_for_causality_update

        self.continuous_actions = False  # self.env.continuous_actions

        self.features = None
        self.dict_for_causality = None
        self.cd = None
        self.ci = None

        np.random.seed(seed)
        random.seed(42)

        self.epsilon = 1
        self.epsilon_decay = 0.995

    def update(self, obs: tensor = None, action: float = None, reward: float = None, next_obs: Dict = None):
        if self.dict_for_causality is not None:
            if obs is not None and reward is not None and action is not None:
                reward = reward.item() if isinstance(reward, torch.Tensor) else reward
                self._update_dict(obs, reward, action)
        else:
            self._initialize_dict(obs)

        if len(self.dict_for_causality[next(iter(self.dict_for_causality))]) > self.timesteps_for_causality_update:
            # print('update')
            dict_detached = detach_dict(self.dict_for_causality)
            df_causality = pd.DataFrame(dict_detached)

            if self.cd is None:
                self.cd = CausalDiscovery(df=df_causality,
                                          dir_name=f'{GLOBAL_PATH_REPO}/navigation/causal_knowledge',
                                          env_name=self.scenario)
            else:
                self.cd.add_data(df_causality)

            self.cd.training()

            causal_graph = self.cd.return_causal_graph()
            df_for_ci = self.cd.return_df()

            if self.ci is None:
                self.ci = CausalInferenceForRL(df_for_ci, causal_graph, self.action_space_size)
            else:
                self.ci.add_data(df_for_ci, causal_graph)

            self._initialize_dict(obs)

    def _initialize_dict(self, observation):
        num_sensors = len(observation) - 6  # Subtracting 6 for PX, PY, VX, VY, DX, DY
        self.features = [f"agent_{self.agent_id}_{feature}" for feature in
                         ['PX', 'PY', 'VX', 'VY', 'DX', 'DY'] + [f'sensor{N}' for N in range(num_sensors)]]
        self.features.append(f"agent_{self.agent_id}_reward")
        self.features.append(f"agent_{self.agent_id}_action")

        self.dict_for_causality = {column: [] for column in self.features}

    def _update_dict(self, observation, reward, action):
        agent_obs = observation.cpu().numpy()
        agent_reward = reward
        agent_action = action if self.continuous_actions else int(action)

        for i, feature in enumerate(self.features[:-2]):
            self.dict_for_causality[feature].append(agent_obs[i])
        self.dict_for_causality[f"agent_{self.agent_id}_reward"].append(agent_reward)
        self.dict_for_causality[f"agent_{self.agent_id}_action"].append(agent_action)

        # print(len(self.dict_for_causality[next(iter(self.dict_for_causality))]))

    def choose_action(self, obs: Dict = None):
        if self.ci is None:
            action = self._random_action_choice()
        else:
            rewards_actions_values = self.ci.get_rewards_actions_values(obs, True)
            action_chosen = self._epsilon_greedy_choice(rewards_actions_values)
            action = torch.tensor([action_chosen], device=self.device)  # Ensure action is wrapped in a list and tensor

        self.epsilon *= self.epsilon_decay

        return action

    def _epsilon_greedy_choice(self, reward_action_values: Dict):
        if not reward_action_values:  # Check if the dictionary is empty
            print('random causal')
            return self._random_action_choice()

        if random.random() < self.epsilon:
            print('causal exploration')
            random_action = random.choice(list(reward_action_values.keys()))
            return torch.tensor([random_action], device=self.device)
        else:
            print('causal exploitation')
            chosen_action = max(reward_action_values, key=reward_action_values.get)
            return torch.tensor([chosen_action], device=self.device)

    def _random_action_choice(self):
        action = torch.randint(
            low=0,
            high=self.action_space_size,
            size=(1,),
            device=self.device,
            dtype=torch.long  # Ensure dtype is specified as long
        )

        return action
