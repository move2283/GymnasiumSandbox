import numpy as np
from gymnasium import spaces

class MyEnvironment:
    def __init__(self):
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(0, 1, shape=(2,))

    def sample_action(self):
        return self.action_space.sample()

    def is_action_valid(self, action):
        return self.action_space.contains(action)

    def sample_observation(self):
        return self.observation_space.sample()

    def is_observation_valid(self, observation):
        return self.observation_space.contains(observation)

if __name__ == "__main__":
    env = MyEnvironment()

    # Sample an action
    action = env.sample_action()
    print(f"Sampled action: {action}")

    # Check if action is valid
    is_valid_action = env.is_action_valid(action)
    print(f"Is action valid? {is_valid_action}")

    # Sample an observation
    observation = env.sample_observation()
    print(f"Sampled observation: {observation}")

    # Check if observation is valid
    is_valid_observation = env.is_observation_valid(observation)
    print(f"Is observation valid? {is_valid_observation}")
