import torch
from vmas import make_env
from vmas.simulator.environment import Wrapper

N_ENVIRONMENTS = 1
N_AGENTS = 2
N_TRAINING_STEPS = 10000


class VMASEnvironment:
    def __init__(self,
                 algorithm=None,
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
                 visualize_render: bool = False,
                 wrapper: Wrapper = None,
                 ):
        self.algorithm = algorithm
        self.if_render = render
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
            y_semidim=self.y_semidim,
            max_steps=100
        )

        return self.env


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
