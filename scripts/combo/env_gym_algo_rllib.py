import torch
import os

has_gpu = torch.cuda.is_available()
os.environ["RLLIB_NUM_GPUS"] = "1" if has_gpu else "0"

import gymnasium as gym
import ray
from ray import tune
from scripts.utils import dynamic_import

num_workers = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class EnvGymAlgoRLlib:
    def __init__(self, env_config, algo_config):
        ray.init(ignore_reinit_error=True)
        self.env_config = env_config
        self.algo_config = algo_config
        self.env = self.create_environment(env_config)
        self.algo_name = self.create_algorithm()

    def create_environment(self, config):
        env = gym.make(config['name'], **config.get('config_env', {}))
        for wrapper in config.get('wrappers', {}).values():
            module, class_name = wrapper['wrapper'].rsplit('.', 1)
            WrapperClass = dynamic_import(module, class_name)
            env = WrapperClass(env, **wrapper.get('kwargs', {}))
        return env

    def create_algorithm(self):
        algo_name = self.algo_config['name'].upper()
        return algo_name

    def train(self, training_episode, seed):
        RLLIB_NUM_GPUS = int(os.environ.get("RLLIB_NUM_GPUS", "0"))
        num_gpus = 0.001 if RLLIB_NUM_GPUS > 0 else 0
        num_gpus_per_worker = (
            (RLLIB_NUM_GPUS - num_gpus) / (num_workers + 1) if device == "cuda" else 0
        )
        tune.run(
            self.algo_name,
            stop={"training_iteration": training_episode},
            checkpoint_freq=1,
            keep_checkpoints_num=2,
            checkpoint_at_end=True,
            verbose=0,
            checkpoint_score_attr="episode_reward_mean",
            # callbacks=[
            #     WandbLoggerCallback(
            #        project=f"{scenario_name}",
            #        api_key="",
            #    )
            # ],
            config={
                "seed": seed,
                "framework": 'torch' if num_gpus > 0 else 'cpu',
                "env": self.env_config['name'],
                "num_gpus": num_gpus,
                "num_workers": num_workers,
                "num_gpus_per_worker": num_gpus_per_worker,
                }
        )

    def evaluate(self, num_episodes, seed):
        pass

    def __del__(self):
        ray.shutdown()
