import numpy as np
from gymnasium import spaces
import string  # Add this import
from gymnasium.spaces.utils import flatten_space, flatten, flatdim, unflatten  # Add this import
from gymnasium.vector.utils import (
    batch_space,
    concatenate,
    iterate,
    create_empty_array,  # Add this import
    create_shared_memory,  # Add this import
    read_from_shared_memory,  # Add this import
)

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

        # Add Graph space with fixed size
        self.graph_space = spaces.Graph(
            node_space=spaces.Box(low=-100, high=100, shape=(3,)),
            edge_space=spaces.Discrete(3),
            seed=42
        )


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

    def flatten_space_examples(self):
        spaces_to_flatten = [
            ("box_space_1", self.box_space_1),
            ("discrete_space", self.discrete_space),
            ("dict_space_1", self.dict_space_1),
            ("graph_space", self.graph_space),
        ]

        for space_name, space in spaces_to_flatten:
            try:
                flattened_space = flatten_space(space)
                print(f"Flattened {space_name}: {flattened_space}")

                sample = space.sample()
                flattened_sample = flatten(space, sample)
                print(f"Flattened sample of {space_name}: {flattened_sample}")

                unflattened_sample = unflatten(space, flattened_sample)
                print(f"Unflattened sample of {space_name}: {unflattened_sample}")

                flattened_dimensions = flatdim(space)
                print(f"Flat dimensions of {space_name}: {flattened_dimensions}")

            except (NotImplementedError, ValueError) as e:  # Handle ValueError in addition to NotImplementedError
                print(f"Flattening not implemented for {space_name}: {e}")

            print()


def batch_space_examples(env):
    space = env.dict_space_1
    n = 5
    batched_space = batch_space(space, n)
    print(f"Batched space for dict_space_1 with n={n}: {batched_space}")


def concatenate_examples(env):
    space = env.box_space_2
    out = np.zeros((2,) + space.shape, dtype=space.dtype)
    items = [space.sample() for _ in range(2)]
    concatenated_items = concatenate(space, items, out)
    print(f"Concatenated samples for box_space_2: {concatenated_items}")


def iterate_examples(env):
    supported_spaces = (spaces.Box, spaces.MultiBinary)

    for space_name, space in vars(env).items():
        if isinstance(space, supported_spaces):
            items = space.sample()
            iterator = iterate(space, items)

            print(f"Iterated samples for {space_name}:")
            for item in iterator:
                print(item)
            print()
        elif isinstance(space, (spaces.Discrete, spaces.MultiDiscrete, spaces.Dict)):
            print(f"Skipping iteration for {space_name} (Discrete space)")
            print()


def create_empty_array_examples(env):
    space = env.box_space_1
    n = 2
    empty_array = create_empty_array(space, n=n, fn=np.zeros)
    print(f"Empty array for box_space_1 with n={n}: {empty_array}")

def create_shared_memory_examples(env):
    space = env.box_space_1
    n = 2
    shared_memory = create_shared_memory(space, n=n)
    print(f"Shared memory for box_space_1 with n={n}: {shared_memory}")

def read_from_shared_memory_examples(env):
    space = env.box_space_1
    n = 2
    shared_memory = create_shared_memory(space, n=n)
    observations = read_from_shared_memory(space, shared_memory, n=n)
    print(f"Observations from shared memory for box_space_1 with n={n}: {observations}")


if __name__ == "__main__":
    env = MyEnvironment()

    for space_name, space in vars(env).items():
        if isinstance(space, spaces.Space):
            env.print_info(space_name)
            print()

    env.flatten_space_examples()

    # Add examples for batch_space, concatenate, and iterate
    batch_space_examples(env)
    concatenate_examples(env)
    iterate_examples(env)

    # Add examples for create_empty_array, create_shared_memory, and read_from_shared_memory
    create_empty_array_examples(env)
    create_shared_memory_examples(env)
    read_from_shared_memory_examples(env)
