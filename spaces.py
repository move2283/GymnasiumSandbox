import numpy as np
from gymnasium import spaces
import string  # Add this import

class MyEnvironment:
    def __init__(self):
        self.action_space = spaces.Discrete(3, seed=42)
        self.observation_space = spaces.Box(0, 1, shape=(2,))
        self.discrete_space = spaces.Discrete(3, start=-1, seed=42)
        self.box_space_1 = spaces.Box(low=-1.0, high=2.0, shape=(3, 4), dtype=np.float64)
        self.box_space_2 = spaces.Box(low=np.array([-1.0, -2.0]), high=np.array([2.0, 4.0]), dtype=np.float64)
        self.multibinary_space_1 = spaces.MultiBinary(5, seed=42)
        self.multibinary_space_2 = spaces.MultiBinary([3, 2], seed=42)
        self.multidiscrete_space_1 = spaces.MultiDiscrete([5, 2, 2], seed=42)
        self.multidiscrete_space_2 = spaces.MultiDiscrete(np.array([[1, 2], [3, 4]]), seed=42)
        self.text_space_1 = spaces.Text(max_length=5, seed=42)
        self.text_space_2 = spaces.Text(min_length=1, max_length=10, charset=string.digits, seed=42)

        # Add Dict spaces
        self.dict_space_1 = spaces.Dict({"position": spaces.Box(-1, 1, shape=(2,)), "color": spaces.Discrete(3)},
                                        seed=42)
        self.dict_space_2 = spaces.Dict(
            {
                "ext_controller": spaces.MultiDiscrete([5, 2, 2]),
                "inner_state": spaces.Dict(
                    {
                        "charge": spaces.Discrete(100),
                        "system_checks": spaces.MultiBinary(10),
                        "job_status": spaces.Dict(
                            {
                                "task": spaces.Discrete(5),
                                "progress": spaces.Box(low=0, high=100, shape=()),
                            }
                        ),
                    }
                ),
            }
        )

        # Add Tuple spaces
        self.tuple_space_1 = spaces.Tuple((spaces.Discrete(2), spaces.Box(-1, 1, shape=(2,))), seed=42)
        self.tuple_space_2 = spaces.Tuple((spaces.MultiBinary(3), spaces.MultiDiscrete([5, 2, 2])), seed=42)

        # Add Sequence spaces
        self.sequence_space_1 = spaces.Sequence(spaces.Box(0, 1), seed=2)
        self.sequence_space_2 = spaces.Sequence(spaces.Box(0, 1), seed=0, stack=True)

    def print_info(self, space_name):
        space = getattr(self, space_name)
        samples = [space.sample() for _ in range(10)]

        if space_name in ["action_space", "observation_space", "discrete_space"]:
            is_valid = [space.contains(sample) for sample in samples]
            print(f"Are {space_name[:-6]} samples valid? {is_valid}")

        if isinstance(space, spaces.Box):
            is_bounded = space.is_bounded()
            print(f"Is {space_name} bounded? {is_bounded}")

        if isinstance(space, spaces.Sequence) and space_name == "sequence_space_1":
            fixed_length = 3
            fixed_length_mask = (fixed_length, None)
            samples = [space.sample(mask=fixed_length_mask) for _ in range(10)]
            print(f"10 fixed-length ({fixed_length}) samples from {space_name}: {samples}")
        else:
            print(f"10 samples from {space_name}: {samples}")


if __name__ == "__main__":
    env = MyEnvironment()

    for space_name, space in vars(env).items():
        if isinstance(space, spaces.Space):
            env.print_info(space_name)
            print()
