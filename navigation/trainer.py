from navigation.algos import RandomAgentVMAS, CausalAgentVMAS
from navigation.vmas_env import VMASEnvironment
import torch
from vmas.simulator.environment import Wrapper
from tqdm.auto import tqdm


class VMASTrainer:
    def __init__(self, n_training_steps: int = 1000, n_environments: int = 1, n_agents: int = 4,
                 algo_name: str = 'only_causal', env_wrapper: Wrapper | None = Wrapper.GYM, rendering: bool = False,
                 x_semidim: float = 0.5, y_semidim: float = 0.5):
        self.n_training_steps = n_training_steps
        self.n_environments = n_environments
        self.n_agents = n_agents
        self.env_wrapper = env_wrapper
        self.if_render = rendering
        self.x_semidim = x_semidim
        self.y_semidim = y_semidim

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.gym_wrapper = Wrapper.GYM

        if self.env_wrapper == self.gym_wrapper:
            self.n_environments = 1

        self.env = self.config_env()
        self.algos = self.config_algos(algo_name)

    def config_env(self):
        print('Device: ', self.device)
        vmas_env = VMASEnvironment(
            scenario_name="navigation",
            render=self.if_render,
            # n_training_episodes=self.n_training_steps,
            num_envs=self.n_environments,
            # save_render=False,
            # random_action=True,
            continuous_actions=False,
            device=self.device,
            n_agents=self.n_agents,
            x_semidim=self.x_semidim,
            y_semidim=self.y_semidim,
            shared_rew=False,
            visualize_render=self.if_render,
            wrapper=self.env_wrapper
        )
        env = vmas_env.initialize_env()

        return env

    def _config_algo(self, algo: str, agent_id: int):
        if algo == 'random':
            algo = RandomAgentVMAS(self.env, self.device)
        elif algo == 'only_causal':
            algo = CausalAgentVMAS(self.env, self.device, agent_id)
        # TODO: causality-driven algo
        return algo

    def config_algos(self, algo_name: str):
        return [self._config_algo(algo_name, i) for i in range(self.n_agents)]

    def train(self):
        if self.env_wrapper is None:
            self.native_train()
        elif self.env_wrapper == self.gym_wrapper:
            self.gym_train()

    def gym_train(self):
        # training process
        pbar = tqdm(range(self.n_training_steps), desc='Training...')
        for episode in pbar:
            observations = self.env.reset()
            done = False
            while not done:
                if observations is not None:
                    actions = [self.algos[i].choose_action(observations[f'agent_{i}']) for i in range(self.n_agents)]
                else:
                    actions = [self.algos[i].choose_action() for i in range(self.n_agents)]
                actions = [tensor.item() for tensor in actions]
                next_observations, rewards, done, info = self.env.step(actions)
                if self.if_render:
                    self.env.render()
                for i in range(self.n_agents):
                    if observations is not None:
                        self.algos[i].update(observations[f'agent_{i}'], actions[i], rewards[f'agent_{i}'],
                                             next_observations[f'agent_{i}'])

    def native_train(self):
        pbar = tqdm(range(self.n_training_steps), desc='Training...')
        for episode in pbar:
            observations = self.env.reset()
            dones = [False] * self.n_agents
            cond = any(dones)
            while not cond:
                if observations is not None:
                    if self.n_environments == 1:
                        actions = [self.algos[i].choose_action(observations[f'agent_{i}'][0]) for i in
                                   range(self.n_agents)]
                    else:
                        actions = [
                            [self.algos[i].choose_action(observations[f'agent_{i}'][env_n]) for env_n in
                             range(self.n_environments)]
                            for i in range(self.n_agents)]
                else:
                    if self.n_environments == 1:
                        actions = [self.algos[i].choose_action() for i in range(self.n_agents)]
                    else:
                        actions = [[self.algos[i].choose_action() for _ in range(self.n_environments)] for i in
                                   range(self.n_agents)]

                next_observations, rewards, dones, info = self.env.step(actions)

                if self.if_render:
                    self.env.render()

                for i in range(self.n_agents):
                    if self.n_environments == 1:
                        if observations is not None:
                            self.algos[i].update(observations[f'agent_{i}'][0], actions[i][0],
                                                 rewards[f'agent_{i}'][0],
                                                 next_observations[f'agent_{i}'][0])
                    else:
                        for env_n in range(self.n_environments):
                            if observations is not None:
                                self.algos[i].update(observations[f'agent_{i}'][env_n], actions[i][env_n],
                                                     rewards[f'agent_{i}'][env_n],
                                                     next_observations[f'agent_{i}'][env_n])

                cond = any(dones)


if __name__ == "__main__":
    trainer = VMASTrainer(env_wrapper=None, n_environments=1, rendering=False, n_agents=4)
    trainer.train()
