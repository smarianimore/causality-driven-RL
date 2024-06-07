import gym
from core.abstract_environment import AbstractEnvironment

class GymWrapper(AbstractEnvironment):
    def __init__(self, env_name):
        self.env = gym.make(env_name)

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def render(self, mode='human'):
        self.env.render(mode)

    def close(self):
        self.env.close()

    def get_state_space(self):
        return self.env.observation_space

    def get_action_space(self):
        return self.env.action_space
