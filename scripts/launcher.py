import yaml
from environments.gym_wrapper import GymWrapper
from environments.vmas_wrapper import VMASWrapper
from algorithms.rllib_meta import RLlibAlgorithm
from algorithms.sb3_meta import SB3Algorithm


def main(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    env_lib = config['environment']['library']
    env_name = config['environment']['name']

    if env_lib == 'gym':
        env = GymWrapper(env_name)
    elif env_lib == 'vmas':
        env = VMASWrapper(env_name)
    else:
        raise ValueError(f"Unsupported environment library: {env_lib}")

    algo_lib = config['algorithm']['library']
    algo_name = config['algorithm']['name']
    algo_config = config['algorithm'].get('hyperparameters', {})

    if algo_lib == 'rllib':
        algo = RLlibAlgorithm(env, algo_name, algo_config)
    elif algo_lib == 'sb3':
        algo = SB3Algorithm(env, algo_name, algo_config)
    else:
        raise ValueError(f"Unsupported algorithm library: {algo_lib}")

    # Example usage
    algo.train(episodes=1000)
    results = algo.evaluate(episodes=100)
    algo.save('model_path')
    algo.load('model_path')
    print(f"Evaluation Results: {results}")


if __name__ == "__main__":
    main('config/config.yaml')
