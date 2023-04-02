import numpy as np
from gymnasium import spaces

class MyEnvironment:
    def __init__(self):
        # Discrete action space with 3 possible values
        self.action_space = spaces.Discrete(3)

        # Continuous observation space with shape (2,) and values between 0 and 1
        self.observation_space = spaces.Box(0, 1, shape=(2,))
        self.additional_spaces()

    def additional_spaces(self):
        # Example of Box with identical bound for each dimension, using float64 instead of float32
        self.box_space_1 = spaces.Box(low=-1.0, high=2.0, shape=(3, 4), dtype=np.float64)

        # Example of Box with independent bound for each dimension, using float64 instead of float32
        self.box_space_2 = spaces.Box(low=np.array([-1.0, -2.0]), high=np.array([2.0, 4.0]), dtype=np.float64)

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

    # Sample from the box_space_1
    box_sample_1 = env.box_space_1.sample()
    print(f"Sampled value from box_space_1: {box_sample_1}")

    # Sample from the box_space_2
    box_sample_2 = env.box_space_2.sample()
    print(f"Sampled value from box_space_2: {box_sample_2}")

    # Check if box_space_1 is bounded
    is_box_space_1_bounded = env.box_space_1.is_bounded()
    print(f"Is box_space_1 bounded? {is_box_space_1_bounded}")

    # Check if box_space_2 is bounded below
    is_box_space_2_bounded_below = env.box_space_2.is_bounded(manner="below")
    print(f"Is box_space_2 bounded below? {is_box_space_2_bounded_below}")
