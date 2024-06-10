import gymnasium as gym
from scripts.environments.wrappers import LoggingWrapper


class GymEnvironment:
    def __init__(self, config):
        self.env_name = config['name']
        self.env = gym.make(self.env_name, **config.get('config_env', {}))
        self.logging = config.get('logging', False)
        self.apply_wrappers(config.get('wrappers', {}))

    def apply_wrappers(self, wrappers):
        for wrapper in wrappers.values():
            module, class_name = wrapper['wrapper'].rsplit('.', 1)
            WrapperClass = getattr(__import__(module, fromlist=[class_name]), class_name)
            self.env = WrapperClass(self.env, **wrapper.get('kwargs', {}))

        self.env = LoggingWrapper(self.env, self.logging)

    def return_env(self):
        return self.env
