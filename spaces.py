import numpy as np
from gymnasium import spaces

class MyEnvironment:
    def __init__(self):
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(0, 1, shape=(2,))
        self.additional_spaces()

    def additional_spaces(self):
        self.multi_binary_space = spaces.MultiBinary(4)
        self.multi_discrete_space = spaces.MultiDiscrete([2, 3, 4])
        self.text_space = spaces.Text(max_length=100)
        self.dict_space = spaces.Dict({
            "action": self.action_space,
            "observation": self.observation_space,
            "multi_binary": self.multi_binary_space,
            "multi_discrete": self.multi_discrete_space,
            "text": self.text_space
        })
        self.tuple_space = spaces.Tuple((self.action_space, self.observation_space))

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
