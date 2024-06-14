from path_repo import GLOBAL_PATH_REPO
"""from scripts.env_algo_combination.env_gym_algo_random import EnvGymAlgoRandom
from scripts.env_algo_combination.env_gym_algo_sb3 import EnvGymAlgoSB3
from scripts.env_algo_combination.env_gym_algo_rllib import EnvGymAlgoRLlib"""
from scripts.env_algo_combination.env_vmas_algo_rllib import EnvVmasAlgoRLlib
from scripts.utils import ConfiguratorParser


class RLFramework:
    def __init__(self, config_file):
        self.logging = None
        self.config_parser = ConfiguratorParser(config_file)
        self.env_algo_instance = self.define_env_algo_instance()
        self.sim_config = self.config_parser.get_simulation_config()

    def define_env_algo_instance(self):
        env_config = self.config_parser.get_environment_config()
        algo_config = self.config_parser.get_algorithm_config()
        sim_config = self.config_parser.get_simulation_config()

        self.logging = env_config['logging']

        env_library = env_config['library'].lower()
        algo_library = algo_config['library'].lower()

        """        if env_library == 'gym' and algo_library == 'sb3':
            return EnvGymAlgoSB3(env_config, algo_config, sim_config)
        elif env_library == 'gym' and algo_library == 'rllib':
            return EnvGymAlgoRLlib(env_config, algo_config, sim_config)"""
        if env_library == 'vmas' and algo_library == 'rllib':
            return EnvVmasAlgoRLlib(env_config, algo_config, sim_config)
        elif env_library == 'vmas' and algo_library == 'sb3':
            pass
            # return EnvVmasAlgoSB3(env_config, algo_config, sim_config)
        """elif env_library == 'gym' and algo_library == 'gym' and algo_config['name'] == 'random':
            return EnvGymAlgoRandom(env_config, algo_config, sim_config)"""

    def train_and_evaluate(self):
        self.env_algo_instance.train()
        print("Training finished.")

        self.env_algo_instance.evaluate()
        print("Evaluation finished.")

        self.env_algo_instance.env.close()


if __name__ == "__main__":
    file_config = f"{GLOBAL_PATH_REPO}/config/config_vmas_rllib_causality.yaml"
    framework = RLFramework(file_config)
    framework.train_and_evaluate()
