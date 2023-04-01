import gymnasium as gym
from gymnasium import Env, spaces
import numpy as np


class CustomEnv(Env):
    def __init__(self):
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=0, high=1, shape=(1,), dtype=float)
        self.seed()

    def step(self, action):
        observation = self.observation_space.sample()
        reward = 1 if action == 0 else -1
        terminated = False
        truncated = False
        info = {}
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        observation = self.observation_space.sample()
        info = {}
        return observation, info

    def seed(self, seed=None):
        self.np_random, _ = gym.utils.seeding.np_random(seed)


gym.register(
    id="CustomEnv-v0",
    entry_point=CustomEnv,
    max_episode_steps=10,
    order_enforce=True,
)

env = gym.make("CustomEnv-v0")

env_spec = gym.spec("CustomEnv-v0")
print(f"Environment spec: {env_spec}")

gym.pprint_registry()

observation, info = env.reset()
for _ in range(10):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    print(f"Observation: {observation}, Reward: {reward}")

env.close()
