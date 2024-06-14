"""import gym
from gym import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(18,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)  # Example action space

    def reset(self):
        return np.zeros(18, dtype=np.float32)

    def step(self, action):
        obs = np.random.randn(18).astype(np.float32)
        reward = 1.0
        done = False
        info = {}
        return obs, reward, done, info

    def render(self, mode='human'):
        pass

env = CustomEnv()
env = DummyVecEnv([lambda: env])

model = PPO('MlpPolicy', env, verbose=0)
model.learn(total_timesteps=1000, progress_bar=True)
model.save("ppo_custom_env")"""
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Ensure the environment is correctly instantiated
env_name = 'CartPole-v1'  # Replace with your environment name or custom environment

# Create the environment
env = gym.make(env_name)
print(type(env))
# Use DummyVecEnv to vectorize the environment
vec_env = DummyVecEnv([lambda: env])
print(type(env))
# Define the RL algorithm
model = PPO('MlpPolicy', vec_env, verbose=0)

# Train the model
model.learn(total_timesteps=100, progress_bar=True)

# Save the model
# model.save("ppo_model")