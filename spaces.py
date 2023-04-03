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
        self.sequence_space_3 = spaces.Sequence(spaces.Discrete(10), seed=42)
        self.sequence_space_4 = spaces.Sequence(spaces.MultiBinary(4), seed=42)
        self.sequence_space_5 = spaces.Sequence(spaces.Dict({"a": spaces.Discrete(2), "b": spaces.Box(-1, 1, shape=(1,))}), seed=42)
        self.sequence_space_6 = spaces.Sequence(spaces.Tuple((spaces.Discrete(2), spaces.Box(-1, 1, shape=(2,)))), seed=42)
        self.sequence_space_7 = spaces.Sequence(spaces.Text(min_length=1, max_length=5, charset=string.ascii_letters), seed=42)
        self.sequence_space_8 = spaces.Sequence(spaces.MultiDiscrete([5, 2]), seed=42, stack=True)
        self.sequence_space_9 = spaces.Sequence(spaces.Box(0, 1), seed=42, stack=True)
        self.sequence_space_10 = spaces.Sequence(spaces.Discrete(10), seed=42)

        # Add Graph space
        self.graph_space = spaces.Graph(node_space=spaces.Box(low=-100, high=100, shape=(3,)), edge_space=spaces.Discrete(3), seed=42)

    def print_info(self, space_name):
        space = getattr(self, space_name)
        samples = [space.sample() for _ in range(10)]

        if space_name in ["action_space", "observation_space", "discrete_space"]:
            is_valid = [space.contains(sample) for sample in samples]
            print(f"Are {space_name[:-6]} samples valid? {is_valid}")

        if isinstance(space, spaces.Box):
            is_bounded = space.is_bounded()
            print(f"Is {space_name} bounded? {is_bounded}")

        if isinstance(space, spaces.Graph):
            num_nodes = 10
            num_edges = None
            sample = space.sample(num_nodes=num_nodes, num_edges=num_edges)
            print(f"Sample graph from {space_name} with {num_nodes} nodes and random number of edges:")

        if isinstance(space, spaces.Sequence):
            # Generate variable-length samples
            mask = (None, None)
            samples_variable = [space.sample(mask=mask) for _ in range(10)]

            # Generate fixed-length samples
            fixed_length = 3
            mask_fixed_length = (fixed_length, None)
            samples_fixed_length = [space.sample(mask=mask_fixed_length) for _ in range(10)]

            print(f"10 variable-length samples from {space_name}: {samples_variable}")
            print(f"10 fixed-length ({fixed_length}) samples from {space_name}: {samples_fixed_length}")
        else:
            print(f"10 samples from {space_name}: {samples}")


if __name__ == "__main__":
    env = MyEnvironment()

    for space_name, space in vars(env).items():
        if isinstance(space, spaces.Space):
            env.print_info(space_name)
            print()
