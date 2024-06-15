from typing import Dict
import random
import pandas as pd
import torch
from torch import tensor
from vmas.simulator.environment import Environment
from navigation.causality_algos import CausalInferenceForRL, CausalDiscovery
from path_repo import GLOBAL_PATH_REPO


class RandomAgentVMAS:
    def __init__(self, env, device):
        self.env = env
        self.device = device

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
    def __init__(self, env: Environment, device, agent_id: int = 0, timesteps_for_causality_update: int = 1000):
        self.env = env
        self.action_space_size = self.env.action_space['agent_0'].n
        self.device = device
        self.agent_id = agent_id
        self.scenario = 'navigation' + f'_agent_{self.agent_id}'
        self.timesteps_for_causality_update = timesteps_for_causality_update

        self.continuous_actions = False  # self.env.continuous_actions

        self.df_causality = None
        self.cd = None
        self.ci = None

        self.epsilon = 1
        self.epsilon_decay = 0.995

    def choose_action(self, obs: Dict = None):
        if self.ci is None:
            action = self.random_action_choice()
        else:
            rewards_actions_values = self.ci.get_rewards_actions_values(obs, True)
            action_chosen = self.epsilon_greedy_choice(rewards_actions_values)
            action = torch.tensor(action_chosen, device=self.device)

        return action

    def update(self, obs: tensor = None, action: float = None, reward: float = None, next_obs: Dict = None):
        if self.df_causality is not None:
            if obs is not None and reward is not None and action is not None:
                self.update_dataframe(obs, reward, action)
        else:
            self.initialize_dataframes(obs)

        if len(self.df_causality) > self.timesteps_for_causality_update:
            if self.cd is None:
                self.cd = CausalDiscovery(df=self.df_causality,
                                          dir_name=f'{GLOBAL_PATH_REPO}/navigation/causal_knowledge',
                                          env_name=self.scenario)
            else:
                self.cd.add_data(self.df_causality)

            self.cd.training()

            causal_graph = self.cd.return_causal_graph()
            df = self.cd.return_df()

            if self.ci is None:
                self.ci = CausalInferenceForRL(df, causal_graph, self.action_space_size)
            else:
                self.ci.add_data(df, causal_graph)

            self.initialize_dataframes(obs)

    def initialize_dataframes(self, observation):
        # Initialize the DataFrames based on the first observation
        columns = []

        self.num_sensors = len(observation) - 6  # Subtracting 6 for PX, PY, VX, VY, DX, DY
        agent_columns = [f"agent_{self.agent_id}_{feature}" for feature in
                         ['PX', 'PY', 'VX', 'VY', 'DX', 'DY'] + [f'sensor{N}' for N in range(self.num_sensors)]]
        agent_columns.append(f"agent_{self.agent_id}_reward")
        agent_columns.append(f"agent_{self.agent_id}_action")
        columns.extend(agent_columns)

        self.df_causality = pd.DataFrame(columns=columns)

    def update_dataframe(self, observation, reward, action):
        data = []
        agent_obs = observation.cpu().numpy()
        agent_reward = reward
        agent_action = action if self.continuous_actions else int(action)

        data.extend(agent_obs)
        data.append(agent_reward)
        data.append(agent_action)
        self.df_causality.loc[len(self.df_causality)] = data

        print(len(self.df_causality))

    def epsilon_greedy_choice(self, reward_action_values: Dict):
        if not reward_action_values:
            raise ValueError("Input dictionary is empty.")

        if reward_action_values == {}:
            return self.random_action_choice()

        # Decide whether to explore or exploit
        if random.random() < self.epsilon:
            # Exploration: choose a random key
            random_action = random.choice(list(reward_action_values.keys()))

            # Convert the chosen key to a tensor and specify the device
            return torch.tensor(random_action, device=self.device)
        else:
            # Exploitation: choose the key with the highest value
            return torch.tensor(max(reward_action_values, key=reward_action_values.get), self.device)

    def random_action_choice(self):
        action = torch.randint(
            low=0,
            high=9,
            size=(1,),
            device=self.device,
        )

        return action
