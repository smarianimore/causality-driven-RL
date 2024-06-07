import yaml
from environments.gym_wrapper.py import GymWrapper
from environments.vmas_wrapper.py import VMASWrapper
from algorithms.rllib_meta import RLLibPPOAlgorithm
from algorithms.sb3_meta import SB3PPOAlgorithm


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
        if algo_name == 'PPO':
            algo = RLLibPPOAlgorithm(env, algo_config)
        else:
            raise ValueError(f"Unsupported RLlib algorithm: {algo_name}")
    elif algo_lib == 'sb3':
        if algo_name == 'PPO':
            algo = SB3PPOAlgorithm(env, algo_config)
        else:
            raise ValueError(f"Unsupported SB3 algorithm: {algo_name}")
    else:
        raise ValueError(f"Unsupported algorithm library: {algo_lib}")

    # Example usage
    algo.train(episodes=1000)
    results = algo.evaluate(episodes=100)
    algo.save('model_path')
    algo.load('model_path')
    print(f"Evaluation Results: {results}")


if __name__ == "__main__":
    main('config/config_launcher.yaml.yaml')
