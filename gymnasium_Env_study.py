import gymnasium as gym

class LunarLanderSimulator:
    def __init__(self, seed=42):
        self.env = gym.make("LunarLander-v2", render_mode="human")
        self.env.action_space.seed(seed)
        self.seed = seed

    def print_env_info(self):
        print("Action space:", self.env.action_space)
        print("Observation space:", self.env.observation_space)
        print("Reward range:", self.env.reward_range)
        print("Environment spec:", self.env.spec)
        print("Environment metadata:", self.env.metadata)

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

    def close(self):
        self.env.close()


if __name__ == "__main__":
    simulator = LunarLanderSimulator()
    simulator.print_env_info()
    simulator.run_simulation()
    simulator.close()
