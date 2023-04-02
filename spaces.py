import numpy as np
from gymnasium import spaces

class MyEnvironment:
    def __init__(self):
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(0, 1, shape=(2,))
        self.box_space_1 = spaces.Box(low=-1.0, high=2.0, shape=(3, 4), dtype=np.float64)
        self.box_space_2 = spaces.Box(low=np.array([-1.0, -2.0]), high=np.array([2.0, 4.0]), dtype=np.float64)

def print_info(env, space_name):
    space = getattr(env, space_name)
    sample = space.sample()
    print(f"Sampled value from {space_name}: {sample}")

    if space_name in ["action_space", "observation_space"]:
        is_valid = space.contains(sample)
        print(f"Is {space_name[:-6]} valid? {is_valid}")

    if isinstance(space, spaces.Box):
        is_bounded = space.is_bounded()
        print(f"Is {space_name} bounded? {is_bounded}")

if __name__ == "__main__":
    env = MyEnvironment()

    for space_name in ["action_space", "observation_space", "box_space_1", "box_space_2"]:
        print_info(env, space_name)
        print()
