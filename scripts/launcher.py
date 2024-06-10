import yaml
from environments.gym_wrapper import GymWrapper
from environments.vmas_wrapper import VMASWrapper
from algorithms.rllib_meta import RLlibAlgorithm
from algorithms.sb3_meta import SB3Algorithm
from path_repo import GLOBAL_PATH_REPO


def main(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    env_lib = config['environment']['library']
    env_name = config['environment']['name']
    env_config = config['environment'].get('config', {})

    if env_lib == 'gym':
        environment_definition = GymWrapper(env_name, env_config)
        env = environment_definition.return_env()
    elif env_lib == 'vmas':
        environment_definition = VMASWrapper(env_name, env_config)
        env = environment_definition.return_env()
    else:
        raise ValueError(f"Unsupported environment library: {env_lib}")

    algo_lib = config['algorithm']['library']
    algo_name = config['algorithm']['name']
    algo_config = config['algorithm'].get('hyperparameters', {})
    seed = config.get('seed', 42)  # Default seed if not provided

    if algo_lib == 'rllib':
        algo = RLlibAlgorithm(env, algo_name, algo_config)
    elif algo_lib == 'sb3':
        algo = SB3Algorithm(env, algo_name, algo_config, seed)
    else:
        raise ValueError(f"Unsupported algorithm library: {algo_lib}")

    algo.train()
    results = algo.evaluate()
    algo.save(f'{env_name}_{algo_name}')

    # Retrieve logs from the LoggingWrapper
    if hasattr(env, 'get_logs'):
        logs = env.get_logs()
    else:
        logs = env.unwrapped.get_logs()
    print(f"Evaluation Results: {results}")
    # print(f"Logs: {logs}")


if __name__ == "__main__":
    main(f'{GLOBAL_PATH_REPO}/config/config_launcher.yaml')
