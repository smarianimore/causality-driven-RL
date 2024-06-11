from path_repo import GLOBAL_PATH_REPO
from scripts.combo.env_gym_algo_sb3 import EnvGymAlgoSB3
from scripts.combo.env_gym_algo_rllib import EnvGymAlgoRLlib
# from scripts.combo.env_vmas_algo_rllib import EnvVmasAlgoRLlib
from scripts.utils import ConfiguratorParser


class RLFramework:
    def __init__(self, config_file):
        self.config_parser = ConfiguratorParser(config_file)
        self.env_algo_instance = self.define_env_algo_instance()
        self.sim_config = self.config_parser.get_simulation_config()

    def define_env_algo_instance(self):
        env_config = self.config_parser.get_environment_config()
        algo_config = self.config_parser.get_algorithm_config()
        env_library = env_config['library'].lower()
        algo_library = algo_config['library'].lower()

        if env_library == 'gym' and algo_library == 'sb3':
            return EnvGymAlgoSB3(env_config, algo_config)
        elif env_library == 'gym' and algo_library == 'rllib':
            return EnvGymAlgoRLlib(env_config, algo_config)
        elif env_library == 'vmas' and algo_library == 'rllib':
            pass
            #return EnvVmasAlgoRLlib(env_config, algo_config)

    def train_and_evaluate(self):
        print("Training started...")
        self.env_algo_instance.train(self.sim_config['training_episodes'], self.sim_config['seed'])
        print("Training finished.")
        print("Evaluation started...")
        self.env_algo_instance.evaluate(self.sim_config['evaluation_episodes'], self.sim_config['seed'])
        print("Evaluation finished.")
        self.env_algo_instance.env.close()


if __name__ == "__main__":
    file_config = f"{GLOBAL_PATH_REPO}/config/config_gym_rllib.yaml"
    framework = RLFramework(file_config)
    framework.train_and_evaluate()
