import gymnasium as gym

class LunarLanderSimulator:
    def __init__(self, seed=42, render_mode=None):
        self.env = gym.make("LunarLander-v2", render_mode=render_mode)  # 当render_mode为"human"时，环境会在每次调用step()方法时自动渲染。因此，在这个特定的情况下，不需要显式调用env.render()。
        self.env.action_space.seed(seed)
        self.seed = seed

    def print_env_info(self):
        print("Action space:", self.env.action_space)
        print("Observation space:", self.env.observation_space)
        print("Observation space high:", self.env.observation_space.high)
        print("Observation space low:", self.env.observation_space.low)
        print("Reward range:", self.env.reward_range)
        print("Environment spec:", self.env.spec)
        print("Environment metadata:", self.env.metadata)
        print("Environment render mode:", self.env.render_mode)
        print("Environment np_random:", self.env.np_random)

    def get_unwrapped_env(self):
        return self.env.unwrapped

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed = seed
        observation, info = self.env.reset(seed=self.seed, options=options)
        return observation, info

    def run_simulation(self, num_steps=1000):
        observation, info = self.reset()

        for _ in range(num_steps):
            action = self.env.action_space.sample()
            observation, reward, terminated, truncated, info = self.env.step(action)

            print(f"Observation: {observation}")
            print(f"Reward: {reward}")
            print(f"Terminated: {terminated}")
            print(f"Truncated: {truncated}")
            print(f"Info: {info}")

            if terminated or truncated:
                observation, info = self.reset()

            self.render()

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()


if __name__ == "__main__":
    simulator = LunarLanderSimulator(render_mode="human")
    simulator.print_env_info()
    simulator.run_simulation()
    unwrapped_env = simulator.get_unwrapped_env()
    print("Unwrapped environment:", unwrapped_env)
    simulator.close()
