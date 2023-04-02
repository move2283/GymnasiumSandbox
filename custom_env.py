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

class CustomEnvRunner:
    def __init__(self, env_id="CustomEnv-v0", max_episode_steps=10):
        with gym.envs.registration.namespace(None):
            gym.register(
                id=env_id,
                entry_point=CustomEnv,
                max_episode_steps=max_episode_steps,
                order_enforce=True,
                autoreset=True,
                disable_env_checker=False,
                reward_threshold=10,
                nondeterministic=True,
            )
        self.env = gym.make(env_id)

    def run(self, num_episodes=1):
        env_spec = gym.spec(self.env.spec.id)
        print(f"Environment spec: {env_spec}")
        gym.pprint_registry()

        # Using get_env_id, parse_env_id and find_highest_version functions
        env_id = gym.envs.registration.get_env_id('custom', 'CustomEnv', 0)
        print(f"Constructed environment id: {env_id}")
        ns, name, version = gym.envs.registration.parse_env_id(env_id)
        print(f"Parsed environment id: Namespace: {ns}, Name: {name}, Version: {version}")
        highest_version = gym.envs.registration.find_highest_version(ns, name)
        print(f"Highest registered version: {highest_version}")

        for episode in range(num_episodes):
            print(f"Episode: {episode + 1}")
            observation, info = self.env.reset()
            for _ in range(10):
                action = self.env.action_space.sample()
                observation, reward, terminated, truncated, info = self.env.step(action)
                print(f"Observation: {observation}, Reward: {reward}")
                if terminated or truncated:
                    break

    def close(self):
        self.env.close()

runner = CustomEnvRunner()
runner.run()
runner.close()
