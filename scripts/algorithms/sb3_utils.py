import numpy as np
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback


class StopTrainingOnEpisodes(BaseCallback):
    def __init__(self, max_episodes: int, verbose=0):
        super(StopTrainingOnEpisodes, self).__init__(verbose)
        self.max_episodes = max_episodes
        self.episode_count = 0

    def _on_step(self) -> bool:
        done = self.locals["dones"]
        if any(done):
            self.episode_count += 1
            if self.verbose > 0:
                print(f"Episode {self.episode_count}")
        return self.episode_count < self.max_episodes

    def reset(self):
        self.episode_count = 0


class FinalPerformanceCallback(BaseCallback):
    def __init__(self, model_name: str = None, verbose: int = 0):
        super(FinalPerformanceCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_actions = []
        self.current_rewards = []
        self.current_actions = []
        self.model_name = model_name

    def _on_step(self) -> bool:
        try:
            rewards = self.locals['rewards']
            actions = self.locals['actions']
            dones = self.locals['dones']

            for i in range(len(dones)):
                self.current_rewards.append(rewards[i])
                self.current_actions.append(actions[i])

                if dones[i]:
                    if self.current_rewards:
                        self.episode_rewards.append(int(np.sum(self.current_rewards)))
                    if self.current_actions:
                        self.episode_actions.append(len(self.current_actions))

                    if self.verbose > 0:
                        print(
                            f"Episode finished. Total Reward: {np.sum(self.current_rewards)}, Total Actions: {len(self.current_actions)}")

                    self.current_rewards = []
                    self.current_actions = []

            return True
        except (IndexError, KeyError) as e:
            if self.verbose > 0:
                print(f"Error accessing 'rewards', 'actions', or 'dones': {e}")
            return True

    def _on_training_end(self) -> None:
        try:
            mean_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0
            std_reward = np.std(self.episode_rewards) if self.episode_rewards else 0
            mean_action_count = np.mean(self.episode_actions) if self.episode_actions else 0
            std_action_count = np.std(self.episode_actions) if self.episode_actions else 0

            res = f"Training {self.model_name} ended. Rewards: {mean_reward} ± {round(std_reward, 2)}, Actions per episode: {mean_action_count} ± {round(std_action_count, 2)}"
            # print(res)
        except Exception as e:
            if self.verbose > 0:
                print(f"Error during training end: {e}")

    def get_episode_rewards(self) -> list:
        return self.episode_rewards

    def get_episode_actions(self) -> list:
        return self.episode_actions
