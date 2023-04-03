import numpy as np
from gym import spaces

class MyEnvironment:
    def __init__(self):
        self.action_space = spaces.Discrete(3, seed=42)
        self.observation_space = spaces.Box(0, 1, shape=(2,))
        self.discrete_space = spaces.Discrete(3, start=-1, seed=42)
        self.box_space_1 = spaces.Box(low=-1.0, high=2.0, shape=(3, 4), dtype=np.float64)
        self.box_space_2 = spaces.Box(low=np.array([-1.0, -2.0]), high=np.array([2.0, 4.0]), dtype=np.float64)
        self.multibinary_space_1 = spaces.MultiBinary(5, seed=42)
        self.multibinary_space_2 = spaces.MultiBinary([3, 2], seed=42)

def print_info(env, space_name):
    space = getattr(env, space_name)
    samples = [space.sample() for _ in range(10)]
    print(f"10 samples from {space_name}: {samples}")

    if space_name in ["action_space", "observation_space", "discrete_space"]:
        is_valid = [space.contains(sample) for sample in samples]
        print(f"Are {space_name[:-6]} samples valid? {is_valid}")

    if isinstance(space, spaces.Box):
        is_bounded = space.is_bounded()
        print(f"Is {space_name} bounded? {is_bounded}")

if __name__ == "__main__":
    env = MyEnvironment()

    for space_name in ["action_space", "observation_space", "discrete_space", "box_space_1", "box_space_2", "multibinary_space_1", "multibinary_space_2"]:
        print_info(env, space_name)
        print()
