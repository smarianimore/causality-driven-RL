from typing import Dict
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple, deque, defaultdict
from torch import tensor, optim, Tensor
from vmas.simulator.environment import Environment
from navigation.causality_algos import CausalInferenceForRL, CausalDiscovery
from navigation.utils import detach_dict
from path_repo import GLOBAL_PATH_REPO

EXPLORATION_GAME_PERCENT = 0.9


class RandomAgentVMAS:
    def __init__(self, env, device, seed: int = 42):
        self.env = env
        self.name = 'random'
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

    def reset_RL_knowledge(self):
        pass


class CausalAgentVMAS:
    def __init__(self, env: Environment, causality_config: Dict = None, device: str = 'cpu', agent_id: int = 0,
                 seed: int = 42, n_steps: int = 100000):
        self.name = 'completely_causal'
        self.env = env
        self.action_space_size = self.env.action_space['agent_0'].n
        self.device = device
        self.agent_id = agent_id
        self.scenario = 'navigation' + f'_agent_{self.agent_id}'
        self.steps_for_causality_update = int(n_steps / 3)  # causality_config.get('steps_for_update', 1000)

        # TODO: setup offline ci and cd
        self.online_cd = causality_config.get('online_cd', True)
        self.online_ci = causality_config.get('online_ci', True)
        self.df = causality_config.get('online_cd', True)
        self.causal_graph = causality_config.get('online_cd', True)
        # self.causal_table = pd.read_pickle(f'C:\\Users\giova\Documents\Research\cdrl_framework\\navigation\causal_knowledge\\navigation_agent_0\causal_table.pkl')

        self.continuous_actions = False  # self.env.continuous_actions

        self.features = None
        self.obs_features = None
        self.next_obs_features = None
        self.dict_for_causality = None
        self.cd = None
        self.ci = None

        np.random.seed(seed)
        random.seed(42)

        self.start_epsilon = 1.0
        self.epsilon = self.start_epsilon
        self.min_epsilon = 0.05
        self.n_steps = n_steps
        self.epsilon_decay = 1 - (-np.log(self.min_epsilon) / (EXPLORATION_GAME_PERCENT * self.n_steps))

    def update(self, obs: Tensor = None, action: float = None, reward: float = None, next_obs: Tensor = None):
        if self.online_cd:
            if self.dict_for_causality is not None:
                if obs is not None and reward is not None and action is not None and next_obs is not None:
                    action = action if self.continuous_actions else int(action)
                    reward = reward.item() if isinstance(reward, torch.Tensor) else reward
                    self._update_dict(obs, reward, action, next_obs)
            else:
                self._initialize_dict(obs)

            if len(self.dict_for_causality[next(iter(self.dict_for_causality))]) > self.steps_for_causality_update:
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
                    self.ci = CausalInferenceForRL(df_for_ci, causal_graph)
                else:
                    self.ci.add_data(df_for_ci, causal_graph)

                self._initialize_dict(obs)

    def _initialize_dict(self, observation):
        self.features = []
        num_sensors = len(observation) - 6  # Subtracting 6 for PX, PY, VX, VY, DX, DY

        """ self.obs_features = [f"agent_{self.agent_id}_{feature}" for feature in
                             ['PX', 'PY', 'VX', 'VY', 'DX', 'DY'] + [f'sensor{N}' for N in range(num_sensors)]]
        self.features += self.obs_features"""
        self.obs_features = None
        self.features.append(f"agent_{self.agent_id}_reward")
        self.features.append(f"agent_{self.agent_id}_action")
        self.next_obs_features = [f"agent_{self.agent_id}_next_{feature}" for feature in
                                  ['PX', 'PY', 'VX', 'VY', 'DX', 'DY'] + [f'sensor{N}' for N in range(num_sensors)]]
        self.features += self.next_obs_features
        self.dict_for_causality = {column: [] for column in self.features}

    def _update_dict(self, observation, reward, action, next_observation):
        agent_obs = observation.cpu().numpy()
        agent_reward = reward
        agent_action = action
        agent_next_obs = next_observation.cpu().numpy()

        if self.obs_features is not None:
            for i, feature in enumerate(self.obs_features):
                self.dict_for_causality[feature].append(agent_obs[i])
        self.dict_for_causality[f"agent_{self.agent_id}_reward"].append(agent_reward)
        self.dict_for_causality[f"agent_{self.agent_id}_action"].append(agent_action)
        if self.next_obs_features is not None:
            for i, feature in enumerate(self.next_obs_features):
                self.dict_for_causality[feature].append(agent_next_obs[i])

        # print(len(self.dict_for_causality[next(iter(self.dict_for_causality))]))

    def choose_action(self, obs: Dict = None):
        if self.ci is None:
            action = self._random_action_choice()
        else:
            rewards_actions_values = self.ci.get_rewards_actions_values(obs, self.online_ci)
            action_chosen = self._epsilon_greedy_choice(rewards_actions_values)
            action = torch.tensor([action_chosen], device=self.device)  # Ensure action is wrapped in a list and tensor

        self.epsilon *= self.epsilon_decay

        return action

    def _epsilon_greedy_choice(self, reward_action_values: Dict):
        if not reward_action_values:  # Check if the dictionary is empty
            print('random causal')
            return self._random_action_choice()

        if random.uniform(0, 1) < self.epsilon:
            random_action = random.choice(list(reward_action_values.keys()))
            print('causal exploration: ', reward_action_values, torch.tensor([random_action], device=self.device))
            return torch.tensor([random_action], device=self.device)
        else:
            values = list(reward_action_values.values())
            if all(value == values[0] for value in values):
                chosen_action = random.choice(list(reward_action_values.keys()))
            else:
                chosen_action = max(reward_action_values, key=reward_action_values.get)
                print('causal exploitation: ', reward_action_values, torch.tensor([chosen_action], device=self.device))
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

    def reset_RL_knowledge(self):
        self.epsilon = self.start_epsilon


class QLearningAgent:
    def __init__(self, env: Environment, device: str = 'cpu', learning_rate: float = 0.001,
                 discount_factor: float = 0.98, epsilon: float = 1.0,
                 min_epsilon=0.05, n_steps: int = 100000):
        self.name = 'qlearning'
        self.device = device
        self.action_space_size = env.action_space['agent_0'].n
        min_collision_distance = 0.005
        self.rows = int(env.world.x_semidim * (1 / min_collision_distance) * 2)
        self.cols = int(env.world.y_semidim * (1 / min_collision_distance) * 2)

        self.continuous_actions = False  # self.env.continuous_actions

        self.start_epsilon = epsilon

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = self.start_epsilon
        self.min_epsilon = min_epsilon
        self.n_steps = n_steps
        self.epsilon_decay = 1 - (-np.log(self.min_epsilon) / (EXPLORATION_GAME_PERCENT * self.n_steps))

        self.q_table = np.zeros((self.rows, self.cols, self.action_space_size))

    def choose_action(self, state: Tensor):
        stateX = int(state[0].cpu().item())
        stateY = int(state[1].cpu().item())

        # Epsilon-greedy action selection
        if random.uniform(0, 1) < self.epsilon:
            # Exploration: choose a random action
            action = random.choice(range(self.action_space_size))
        else:
            # Exploitation: choose the best action based on current Q-values
            action = np.argmax(self.q_table[stateX, stateY])

        action = torch.tensor([action], device=self.device)
        return action

    def update(self, obs: Tensor = None, action: float = None, reward: float = None, next_obs: Tensor = None):

        reward = reward.item() if isinstance(reward, torch.Tensor) else reward
        action = action if self.continuous_actions else int(action)

        stateX = int(obs[0].cpu().item())
        stateY = int(obs[1].cpu().item())
        next_stateX = int(next_obs[0].cpu().item())
        next_stateY = int(next_obs[1].cpu().item())
        best_next_action = np.argmax(self.q_table[next_stateX, next_stateY])
        td_target = reward + self.discount_factor * self.q_table[next_stateX, next_stateY, best_next_action]
        td_error = td_target - self.q_table[stateX, stateY, action]
        self.q_table[stateX, stateY, action] += self.learning_rate * td_error

        self._decay_epsilon()

    def _decay_epsilon(self):
        # Decay epsilon
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def reset_RL_knowledge(self):
        self.q_table = np.zeros((self.rows, self.cols, self.action_space_size))
        self.epsilon = self.start_epsilon


class DynamicQLearningAgent:
    def __init__(self, env: Environment, device: str = 'cpu', learning_rate: float = 0.001,
                 discount_factor: float = 0.98, epsilon: float = 1.0,
                 min_epsilon=0.05, n_steps: int = 100000):
        self.name = 'dynamic_qlearning'
        self.device = device
        self.action_space_size = env.action_space['agent_0'].n

        self.continuous_actions = False  # self.env.continuous_actions

        self.start_epsilon = epsilon

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = self.start_epsilon
        self.min_epsilon = min_epsilon
        self.n_steps = n_steps
        self.epsilon_decay = 1 - (-np.log(self.min_epsilon) / (EXPLORATION_GAME_PERCENT * self.n_steps))

        self.q_table = defaultdict(lambda: np.zeros(self.action_space_size))

    def choose_action(self, state: Tensor):
        state_tuple = self._state_to_tuple(state)
        if random.uniform(0, 1) < self.epsilon:
            # print('exploration')
            action = random.choice(range(self.action_space_size))
        else:
            # print('exploitation')
            state_action_values = self.q_table[state_tuple]
            action = np.argmax(state_action_values)

        action = torch.tensor([action], device=self.device)
        return action

    def update(self, obs: Tensor = None, action: float = None, reward: float = None, next_obs: Tensor = None):
        action = int(action)
        state_tuple = self._state_to_tuple(obs)
        next_state_tuple = self._state_to_tuple(next_obs)

        best_next_action = np.argmax(self.q_table[next_state_tuple])
        td_target = reward + self.discount_factor * self.q_table[next_state_tuple][best_next_action]
        td_error = td_target - self.q_table[state_tuple][action]
        self.q_table[state_tuple][action] += self.learning_rate * td_error

        self._decay_epsilon()

    def _decay_epsilon(self):
        # Decay epsilon
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def _state_to_tuple(self, state):
        """Convert tensor state to a tuple to be used as a dictionary key."""
        return tuple(state.cpu().numpy())

    def reset_RL_knowledge(self):
        self.q_table = defaultdict(lambda: np.zeros(self.action_space_size))
        self.epsilon = self.start_epsilon


# Define a namedtuple for Experience Replay
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)  # Adjust input_size to match state dimension
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))  # Input x should be (batch_size, input_size)
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent:
    def __init__(self, env: Environment, device='cpu', learning_rate=0.0001, discount_factor=0.98, epsilon=1.0,
                 min_epsilon=0.05, batch_size=64, replay_memory_size=10000, n_steps: int = 100000):
        self.name = 'dqn'
        self.device = device
        self.env = env
        self.action_space_size = env.action_space['agent_0'].n
        self.state_space_size = 18
        self.continuous_actions = False  # Assuming discrete actions

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.n_steps = n_steps
        self.start_epsilon = epsilon
        self.epsilon = self.start_epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = 1 - (-np.log(self.min_epsilon) / (EXPLORATION_GAME_PERCENT * self.n_steps))

        # Q-Network
        self.q_network = QNetwork(self.state_space_size, self.action_space_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # Experience Replay
        self.batch_size = batch_size
        self.len_replay_memory_size = replay_memory_size
        self.replay_memory = deque(maxlen=self.len_replay_memory_size)

    def choose_action(self, state: Tensor):
        # state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        # Epsilon-greedy action selection
        if random.uniform(0, 1) < self.epsilon:
            action = random.randrange(self.action_space_size)
        else:
            with torch.no_grad():
                q_values = self.q_network(state)
                action = torch.argmax(q_values).item()

        action = torch.tensor([action], device=self.device)

        return action

    def update(self, state: Tensor, action: Tensor, reward: Tensor, next_state: Tensor):
        # Store transition in replay memory
        self.replay_memory.append(Transition(state, action, next_state, reward))

        # Sample a random batch from replay memory
        if len(self.replay_memory) < self.batch_size:
            return

        transitions = random.sample(self.replay_memory, self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device,
                                      dtype=torch.bool)
        non_final_next_states = torch.stack(
            [s.clone().to(self.device).detach() for s in batch.next_state if s is not None])
        non_final_next_states = non_final_next_states.to(torch.float)

        state_batch = torch.stack([s.clone().to(self.device).detach() for s in batch.state])
        state_batch = state_batch.to(torch.float)

        action_batch = torch.tensor(batch.action, dtype=torch.int64, device=self.device)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32, device=self.device)

        # Compute Q-values for current and next states
        state_action_values = self.q_network(state_batch).gather(1, action_batch.unsqueeze(1))
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = self.q_network(non_final_next_states).max(1)[0].detach()

        # Compute expected Q-values
        expected_state_action_values = (next_state_values * self.discount_factor) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def save(self, filename):
        torch.save(self.q_network.state_dict(), filename)

    def load(self, filename):
        self.q_network.load_state_dict(torch.load(filename, map_location=self.device))
        self.q_network.eval()

    def reset_RL_knowledge(self):
        # Q-Network
        self.q_network = QNetwork(self.state_space_size, self.action_space_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)

        self.replay_memory = deque(maxlen=self.len_replay_memory_size)
