from navigation.algos import RandomAgentVMAS, CausalAgentVMAS
from navigation.vmas_env import VMASEnvironment
import torch
from vmas.simulator.environment import Wrapper
from tqdm.auto import tqdm

N_TRAINING_STEPS = 1000
N_AGENTS = 2


def config_env():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device: ', device)
    vmas_env = VMASEnvironment(
        scenario_name="navigation",
        render=False,
        n_training_episodes=N_TRAINING_STEPS,
        num_envs=1,
        save_render=False,
        random_action=True,
        continuous_actions=False,
        device=device,
        n_agents=N_AGENTS,
        x_semidim=0.3,
        y_semidim=0.3,
        shared_rew=False,
        visualize_render=False,
        wrapper=Wrapper.GYM
    )
    env = vmas_env.initialize_env()
    return env


def config_algo(env, agent_id):
    device = 'cuda'
    # algo = RandomAgentVMAS(env, N_ENVIRONMENTS, device)
    algo = CausalAgentVMAS(env, device, agent_id)
    return algo


def main():
    env = config_env()

    # define algo
    algos = [config_algo(env, i) for i in range(N_AGENTS)]

    # training process
    pbar = tqdm(range(N_TRAINING_STEPS), desc='Training...')
    for episode in pbar:
        observations = env.reset()
        done = False
        while not done:
            if observations is not None:
                actions = [algos[i].choose_action(observations[f'agent_{i}']) for i in range(N_AGENTS)]
            else:
                actions = [algos[i].choose_action() for i in range(N_AGENTS)]
            actions = [tensor.item() for tensor in actions]
            next_observations, rewards, done, info = env.step(actions)
            env.render()
            for i in range(N_AGENTS):
                if observations is not None:
                    algos[i].update(observations[f'agent_{i}'], actions[i], rewards[f'agent_{i}'],
                                    next_observations[f'agent_{i}'])


if __name__ == "__main__":
    main()
