from navigation.algos import RandomAgentVMAS, CausalAgentVMAS
from navigation.vmas_env import VMASEnvironment
import torch
from vmas.simulator.environment import Wrapper
from tqdm.auto import tqdm

N_TRAINING_STEPS = 1000
N_ENVIRONMENTS = 1
N_AGENTS = 3


def config_env():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device: ', device)
    device = 'cpu'
    vmas_env = VMASEnvironment(
        scenario_name="navigation",
        render=True,
        n_training_episodes=N_TRAINING_STEPS,
        num_envs=N_ENVIRONMENTS,
        save_render=False,
        random_action=True,
        continuous_actions=False,
        device=device,
        n_agents=N_AGENTS,
        x_semidim=0.2,
        y_semidim=0.2,
        shared_rew=False,
        visualize_render=True,
        # wrapper=Wrapper.RLLIB
    )
    env = vmas_env.initialize_env()
    return env


def config_algo(env, agent_id):
    device = env.device
    # algo = RandomAgentVMAS(env, N_ENVIRONMENTS, device)
    algo = CausalAgentVMAS(env, device, agent_id)
    return algo


def main():
    env = config_env()

    # define algo
    algos = [config_algo(env, i) for i in range(N_AGENTS)]
    # training process
    pbar = tqdm(range(N_TRAINING_STEPS), desc='Training...')

    observations = None
    for episode in pbar:
        observations = env.reset()
        dones = [False * N_AGENTS]
        while not all(dones):
            if observations is not None:
                if N_ENVIRONMENTS == 1:
                    actions = [algos[i].choose_action(observations[f'agent_{i}'][0]) for i in range(N_AGENTS)]
                else:
                    actions = [
                        [algos[i].choose_action(observations[f'agent_{i}'][env_n]) for env_n in range(N_ENVIRONMENTS)]
                        for i in range(N_AGENTS)]
            else:
                if N_ENVIRONMENTS == 1:
                    actions = [algos[i].choose_action() for i in range(N_AGENTS)]
                else:
                    actions = [[algos[i].choose_action() for _ in range(N_ENVIRONMENTS)] for i in range(N_AGENTS)]

            next_observations, rewards, dones, info = env.step(actions)
            env.render()

            for i in range(N_AGENTS):
                if N_ENVIRONMENTS == 1:
                    if observations is not None:
                        algos[i].update(observations[f'agent_{i}'][0], actions[i][0], rewards[f'agent_{i}'][0],
                                        next_observations[f'agent_{i}'][0])
                else:
                    for env_n in range(N_ENVIRONMENTS):
                        if observations is not None:
                            algos[i].update(observations[f'agent_{i}'][env_n], actions[i][env_n],
                                            rewards[f'agent_{i}'][env_n],
                                            next_observations[f'agent_{i}'][env_n])


if __name__ == "__main__":
    main()
