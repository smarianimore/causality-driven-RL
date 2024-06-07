import time
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

" ****************************************************************************************************************** "


class StopTrainingOnEpisodesCallback(BaseCallback):
    def __init__(self, num_episodes, verbose=0):
        super(StopTrainingOnEpisodesCallback, self).__init__(verbose)
        self.num_episodes = num_episodes
        self.episode_count = 0
        self.start_time = None
        self.end_time = None

    def _on_training_start(self):
        self.start_time = time.time()

    def _on_step(self) -> bool:
        try:
            dones = self.locals.get('dones')
            self.episode_count += sum(dones)
            return self.episode_count < self.num_episodes
        except (IndexError, KeyError) as e:
            if self.verbose > 0:
                print(f"Error accessing 'dones': {e}")
            return True

    def _on_training_end(self):
        self.end_time = time.time()
        total_time = self.end_time - self.start_time
        # print(f"Total training time: {total_time} seconds")

    def get_total_time(self) -> float:
        return self.end_time - self.start_time if self.end_time else None


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


class FrozenLake_HoleCounterCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(FrozenLake_HoleCounterCallback, self).__init__(verbose)
        self.hole_count = 0

    def _on_step(self) -> bool:
        # Get the current environment state for each environment in the vectorized environment
        for i in range(self.training_env.num_envs):
            env = self.training_env.get_attr('unwrapped')[i]
            state = env.get_wrapper_attr('s')  # self.training_env.get_attr('s')[i]
            if hasattr(env, 'desc'):
                desc = env.get_wrapper_attr('desc')  # self.training_env.get_attr('desc')[i]
                row, col = np.unravel_index(state, desc.shape)
                if desc[row, col] == b'H':
                    self.hole_count += 1
                    print(self.hole_count)
                    if self.verbose > 0:
                        print(
                            f"Agent in env {i} fell into a hole at position ({row}, {col}). Total holes: {self.hole_count}")
        return True

    def _on_training_end(self) -> None:
        res = f"Training ended. The agent fell into holes {self.hole_count} times."
        # print(res)

    def get_hole_count(self) -> int:
        return self.hole_count
