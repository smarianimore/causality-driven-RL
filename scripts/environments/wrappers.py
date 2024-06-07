from typing import Tuple
import gymnasium as gym
import importlib
import numpy as np

LABEL_BOUNDARY = 'border'

LABEL_ORIGINAL_OBS = 'default'
LABEL_NEW_OBS = 'new'
LABEL_CONCAT_OBS = 'concat'


def env_wrappers_manager(env, reward_wrapper=None, observation_wrapper=None):
    if reward_wrapper:
        wrapper_class_path = reward_wrapper['wrapper']
        module_name, class_name = wrapper_class_path.rsplit(".", 1)
        reward_wrapper_class = getattr(importlib.import_module(module_name), class_name)
        wrapper_kwargs = reward_wrapper.get('kwargs', {})
        env = reward_wrapper_class(env, **wrapper_kwargs)

    if observation_wrapper:
        wrapper_class_path = observation_wrapper['wrapper']
        module_name, class_name = wrapper_class_path.rsplit(".", 1)
        observation_wrapper_class = getattr(importlib.import_module(module_name), class_name)
        wrapper_kwargs = observation_wrapper.get('kwargs', {})
        env = observation_wrapper_class(env, **wrapper_kwargs)

    return env


class FrozenLake_ObservationSpaceWrapper(gym.ObservationWrapper):
    def __init__(self, environment, observation_matrix_shape: Tuple[int, int], kind_of_obs: str):
        super().__init__(environment)

        # This is the space for the typical observations from the environment
        self.gym_obs = environment.observation_space
        self.world = np.array(environment.desc.tolist(), dtype='<U10')

        self.entities_world, self.n_entities_world = _count_unique_strings(self.world)

        # add "bound" in entities
        self.entities_world.add(LABEL_BOUNDARY)
        self.n_entities_world += 1

        self.world_rows, self.world_cols = self.world.shape

        self.observation_matrix_shape = observation_matrix_shape
        self.kind_of_obs = kind_of_obs

        # Define a new observation space
        if self.kind_of_obs == LABEL_ORIGINAL_OBS:
            gym_space = {LABEL_ORIGINAL_OBS: self.gym_obs}
        elif self.kind_of_obs == LABEL_NEW_OBS:
            gym_space = _create_feature_dict(self.entities_world, self.observation_matrix_shape)
        elif self.kind_of_obs == LABEL_CONCAT_OBS:
            gym_space = _create_feature_dict(self.entities_world, self.observation_matrix_shape)
            gym_space[LABEL_ORIGINAL_OBS] = self.gym_obs
        else:
            raise AssertionError('kind of observation chosen is not implemented')

        self.observation_space = gym.spaces.Dict(gym_space)

    def observation(self, obs):
        if isinstance(obs, np.ndarray) and obs.ndim > 1:  # Checking if observations are vectorized
            new_observations = [self._handle_single_observation(o) for o in obs]
            return np.array(new_observations)
        else:
            return self._handle_single_observation(obs)

    def _handle_single_observation(self, obs):
        if self.kind_of_obs == LABEL_ORIGINAL_OBS:
            return {LABEL_ORIGINAL_OBS: obs}
        elif self.kind_of_obs == LABEL_NEW_OBS:
            return self._define_new_observations(obs)
        elif self.kind_of_obs == LABEL_CONCAT_OBS:
            new_observation = self._define_new_observations(obs)
            new_observation.update({LABEL_ORIGINAL_OBS: obs})
            return new_observation
        else:
            raise AssertionError('kind of observation chosen is not implemented')

    def _define_new_observations(self, obs: int) -> dict:
        def _get_full_matrix(obs2):
            # Create a copy of the base grid for manipulation
            matrix = np.copy(self.world)
            return matrix

        def get_obs_matrix(obs1):
            start_matrix = _get_full_matrix(obs1)

            row, col = divmod(obs1, self.world_rows)
            # Create a NxN submatrix centered on the agent
            submatrix = np.full(self.observation_matrix_shape, LABEL_BOUNDARY, dtype='<U10')
            for i in range(-1, 2):
                for j in range(-1, 2):
                    sub_row = row + i
                    sub_col = col + j
                    if 0 <= sub_row < 4 and 0 <= sub_col < 4:
                        submatrix[i + 1, j + 1] = start_matrix[sub_row, sub_col]

            return submatrix.tolist()

        obs_matrix = get_obs_matrix(obs)
        features_dict = _new_feature_dict(obs_matrix, self.entities_world)

        ordered_features_dict = {key: features_dict[key] for key in self.entities_world if key in features_dict}

        return ordered_features_dict


class FrozenLake_RewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super(FrozenLake_RewardWrapper, self).__init__(env)

    def reward(self, reward):
        desc = self.env.desc
        # Check the position of the agent
        if desc[self.env.s // self.env.ncol, self.env.s % self.env.ncol] == b'H':
            return -1  # Apply a penalty if the agent steps on a hole
        return reward


class Taxi_ObservationSpaceWrapper(gym.ObservationWrapper):
    def __init__(self, environment, observation_matrix_shape: Tuple, kind_of_obs: str):
        super().__init__(environment)
        self.gym_obs = environment.observation_space
        # self.world = np.array(environment.get_wrapper_attr('desc').tolist(), dtype='<U10')
        self.world = np.array(environment.desc.tolist(), dtype='<U10')
        self.world_rows, self.world_cols = self.world.shape
        # self.locs = environment.get_wrapper_attr('locs')
        self.locs = environment.locs

        self.entities_world, self.n_entities_world = _count_unique_strings(self.world)

        # add "bound" in entities
        self.entities_world.add(LABEL_BOUNDARY)
        self.n_entities_world += 1

        self.world_rows, self.world_cols = self.world.shape

        self.observation_matrix_shape = observation_matrix_shape
        self.kind_of_obs = kind_of_obs

        # Define a new observation space
        if self.kind_of_obs == LABEL_ORIGINAL_OBS:
            gym_space = {LABEL_ORIGINAL_OBS: self.gym_obs}
        elif self.kind_of_obs == LABEL_NEW_OBS:
            gym_space = _create_feature_dict(self.entities_world, self.observation_matrix_shape)
        elif self.kind_of_obs == LABEL_CONCAT_OBS:
            gym_space = _create_feature_dict(self.entities_world, self.observation_matrix_shape)
            gym_space[LABEL_ORIGINAL_OBS] = self.gym_obs
        else:
            raise AssertionError('kind of observation chosen is not implemented')

        self.observation_space = gym.spaces.Dict(gym_space)

    def observation(self, obs):
        if isinstance(obs, np.ndarray) and obs.ndim > 1:  # Checking if observations are vectorized
            new_observations = [self._handle_single_observation(o) for o in obs]
            return np.array(new_observations)
        else:
            return self._handle_single_observation(obs)

    def _handle_single_observation(self, obs):
        if self.kind_of_obs == LABEL_ORIGINAL_OBS:
            return {LABEL_ORIGINAL_OBS: obs}
        elif self.kind_of_obs == LABEL_NEW_OBS:
            return self._define_new_observations(obs)
        elif self.kind_of_obs == LABEL_CONCAT_OBS:
            new_observation = self._define_new_observations(obs)
            new_observation.update({LABEL_ORIGINAL_OBS: obs})
            return new_observation
        else:
            raise AssertionError('kind of observation chosen is not implemented')

    def _define_new_observations(self, obs):
        def custom_decode(i):
            out = []
            out.append(i % 4)  # dest_idx
            i = i // 4

            out.append(i % 5)  # pass_loc
            i = i // 5

            out.append(i % 5)  # taxi_col
            i = i // 5

            out.append(i)  # taxi_row
            assert 0 <= i < 5  # Ensure taxi_row is within the correct range

            x = list(reversed(out))  # This needs to reverse the list to match the order of encode inputs
            return x

        def _get_full_matrix(obs2):
            taxi_row, taxi_col, pass_idx, dest_idx = custom_decode(obs2)
            dynamic_world = np.copy(self.world)  # Copy the static world for modifications

            # Update taxi location with 'T'
            dynamic_world[taxi_row][taxi_col] = 'T'

            # Update passenger and destination markers
            for idx, loc in enumerate(self.locs):
                pr, pc = loc
                if idx == pass_idx:  # Passenger is here
                    dynamic_world[pr][pc] = 'P'
                if idx == dest_idx:  # Destination is here
                    dynamic_world[pr][pc] += 'D'  # Append 'D' to indicate destination
            return dynamic_world

        def get_obs_matrix(obs1):
            matrix = _get_full_matrix(obs1)
            taxi_row, taxi_col, _, _ = custom_decode(obs1)
            submatrix = np.full(self.observation_matrix_shape, ' ', dtype='<U10')
            for i in range(-1, 2):
                for j in range(-1, 2):
                    sub_row = taxi_row + i
                    sub_col = taxi_col + j
                    if 0 <= sub_row < self.world_rows and 0 <= sub_col < self.world_cols:
                        submatrix[i + 1, j + 1] = matrix[sub_row][sub_col]
            return submatrix

        obs_matrix = get_obs_matrix(obs)
        features_dict = _new_feature_dict(obs_matrix, self.entities_world)

        ordered_features_dict = {key: features_dict[key] for key in self.entities_world if key in features_dict}

        return ordered_features_dict


def _count_unique_strings(array):
    unique, counts = np.unique(array, return_counts=True)
    return set(unique), len(unique)


def _create_feature_dict(entities, shape):
    assert all(isinstance(dim, int) for dim in shape), "All elements in shape must be integers"
    return {entity: gym.spaces.Box(low=0, high=1, shape=shape, dtype=np.float32) for entity in entities}


def _new_feature_dict(matrix, entities_world):
    features = {entity: np.zeros_like(matrix, dtype=np.float32) for entity in entities_world}
    for i, row in enumerate(matrix):
        for j, element in enumerate(row):
            if element in features:
                features[element][i, j] = 1
    return features
