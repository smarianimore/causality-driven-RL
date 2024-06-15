import random
import time
import pandas as pd
import torch
from tqdm.auto import tqdm
from vmas import make_env
from vmas.simulator.core import Agent
from vmas.simulator.environment import Wrapper
from vmas.simulator.utils import save_video
import multiprocessing
from navigation.causality_algos import CausalDiscovery, CausalInferenceForRL
from path_repo import GLOBAL_PATH_REPO
from try_ppo import PPO

N_ENVIRONMENTS = 5
N_AGENTS = 2
N_TRAINING_STEPS = 10000


class VMASEnvironment:
    def __init__(self,
                 algorithm = None,
                 render: bool = False,
                 save_render: bool = False,
                 num_envs: int = 32,
                 n_training_episodes: int = 1000,
                 random_action: bool = False,
                 device: str = "cuda",
                 scenario_name: str = "navigation",
                 n_agents: int = 4,
                 x_semidim: float = None,
                 y_semidim: float = None,
                 shared_rew: bool = None,
                 continuous_actions: bool = True,
                 visualize_render: bool = True,
                 wrapper: Wrapper = None,
                 ):
        self.algorithm = algorithm
        self.render = render
        self.save_render = save_render
        self.num_envs = num_envs
        self.n_training_episodes = n_training_episodes
        self.random_action = random_action
        self.device = device
        self.scenario_name = scenario_name
        self.n_agents = n_agents
        self.x_semidim = x_semidim
        self.y_semidim = y_semidim
        self.shared_rew = shared_rew
        self.continuous_actions = continuous_actions
        self.visualize_render = visualize_render
        self.wrapper = wrapper
        self.env = None
        self.frame_list = []
        self.dfs = {}  # Dictionary to hold DataFrames for each environment
        self.columns_initialized = False  # Flag to indicate if columns are initialized

    def _get_deterministic_action(self, agent: Agent):
        if self.continuous_actions:
            action = -agent.action.u_range_tensor.expand(self.env.batch_dim, agent.action_size)
        else:
            action = (torch.tensor([1], device=self.env.device, dtype=torch.long)
                      .unsqueeze(-1)
                      .expand(self.env.batch_dim, 1))
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
            seed=None,
            shared_rew=self.shared_rew,
            n_agents=self.n_agents,
            x_semidim=self.x_semidim,
            y_semidim=self.y_semidim
        )
        print(self.env.observation_space, self.env.action_space)

        print(1/(0.05 * self.x_semidim), 1/(0.05 * self.y_semidim))

        return self.env.observation_space.shape, self.env.action_space.shape

    def initialize_dataframes(self, observations):
        # Initialize the DataFrames based on the first observation
        columns = []
        for agent_id in range(self.n_agents):
            self.num_sensors = observations[f'agent_{agent_id}'].shape[
                                   1] - 6  # Subtracting 6 for PX, PY, VX, VY, DX, DY
            agent_columns = [f"agent_{agent_id}_{feature}" for feature in
                             ['PX', 'PY', 'VX', 'VY', 'DX', 'DY'] + [f'sensor{N}' for N in range(self.num_sensors)]]
            agent_columns.append(f"agent_{agent_id}_reward")
            agent_columns.append(f"agent_{agent_id}_action")
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

    def run(self):
        assert not (self.save_render and not self.render), "To save the video you have to render it"

        self.initialize_env()

        init_time = time.time()
        pbar = tqdm(range(self.n_training_episodes), desc='Training...')

        for _ in pbar:
            dict_actions = random.choice([True, False])
            actions = {} if dict_actions else []
            for agent in self.env.agents:
                if not self.random_action:
                    action = self._get_deterministic_action(agent)
                else:
                    action = self.env.get_random_action(agent)
                if dict_actions:
                    actions.update({agent.name: action})
                else:
                    actions.append(action)

            obs, rews, dones, info = self.env.step(actions)
            # SHAPES: obs [n_envs, 18], rews [n_envs], dones [n_envs]

            if not self.columns_initialized:
                self.initialize_dataframes(obs)

            # Update the DataFrames for each environment
            for env_id in range(self.num_envs):
                self.update_dataframe(obs, rews, actions, env_id)

            if self.render:
                frame = self.env.render(
                    mode="rgb_array",
                    agent_index_focus=None,
                    visualize_when_rgb=self.visualize_render,
                )
                if self.save_render:
                    self.frame_list.append(frame)

        total_time = time.time() - init_time
        print(
            f"It took: {total_time}s for {self.n_training_episodes} steps of {self.num_envs} parallel environments on device {self.device} "
            f"for {self.scenario_name} scenario.")

        if self.render and self.save_render:
            save_video(self.scenario_name, self.frame_list, fps=1 / self.env.scenario.world.dt)

    def _concat_dfs(self):
        agent_dataframes = {f"agent{agent_idx}": pd.DataFrame() for agent_idx in range(N_AGENTS)}

        for agent_idx in range(N_AGENTS):
            agent_columns = [f'agent_{agent_idx}_PX', f'agent_{agent_idx}_PY', f'agent_{agent_idx}_VX',
                             f'agent_{agent_idx}_VY',
                             f'agent_{agent_idx}_DX', f'agent_{agent_idx}_DY'] + \
                            [f'agent_{agent_idx}_sensor{i}' for i in range(self.num_sensors)] + \
                            [f'agent_{agent_idx}_reward', f'agent_{agent_idx}_action']

            agent_df_list = [df[agent_columns] for df in self.dfs.values() if isinstance(df, pd.DataFrame)]
            concatenated_df = pd.concat(agent_df_list, ignore_index=True)

            agent_dataframes[f"agent{agent_idx}"] = concatenated_df

        return agent_dataframes

    def causality_extraction(self):
        #agent_dataframes = {'agent_0_vecchio': pd.read_pickle(
            #'C:\\Users\giova\Documents\Research\cdrl_framework\\navigation\causal_knowledge\\agent0\df_original.pkl')}
        # 'agent_1': pd.read_pickle('C:\\Users\giova\Documents\Research\cdrl_framework\\navigation\causal_knowledge\\agent1\df_original.pkl')}

        agent_dataframes = self._concat_dfs()
        n = N_AGENTS if N_AGENTS < multiprocessing.cpu_count() else multiprocessing.cpu_count()
        with multiprocessing.Pool(processes=n) as pool:
            causal_graphs, dfs = pool.starmap(self.cd_chunk,
                                              [(agent_df, agent_name) for agent_name, agent_df in
                                               agent_dataframes.items()])
        n_graphs = len(causal_graphs)
        tasks = []
        for i in range(n_graphs):
            tasks.append((dfs[i], causal_graphs[i], agent_dataframes[i]))

        with multiprocessing.Pool(processes=n) as pool:
            causal_tables = pool.starmap(self.ci_chunk, tasks)

    @staticmethod
    def cd_chunk(df, env_name):
        # try:
        filtered_df = df.loc[:, df.std() != 0]
        dir_name = f'{GLOBAL_PATH_REPO}/navigation/causal_knowledge'

        cd = CausalDiscovery(filtered_df, dir_name, env_name)
        cd.training()

        return cd.return_causal_graph(), cd.return_df()
        """except Exception as e:
            print(f'CD failed in {env_name}')
            print(e)"""

    @staticmethod
    def ci_chunk(df, causal_graph, env_name):
        dir_name = f'{GLOBAL_PATH_REPO}/navigation/causal_knowledge'
        ci = CausalInferenceForRL(df, causal_graph, dir_save=dir_name, name_save=env_name)
        causal_table = ci.create_causal_table()

        return causal_table


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device: ', device)
    vmas_env = VMASEnvironment(
        scenario_name="navigation",
        render=False,
        n_training_episodes=N_TRAINING_STEPS,
        num_envs=N_ENVIRONMENTS,
        save_render=False,
        random_action=True,
        continuous_actions=False,
        device=device,
        n_agents=N_AGENTS,
        x_semidim=0.5,
        y_semidim=0.5,
        shared_rew=False,
        visualize_render=False,
        # wrapper=Wrapper.GYM
    )

    vmas_env.run()
    vmas_env.causality_extraction()
