import gymnasium as gym

class LunarLanderSimulator:
    def __init__(self, seed=42, render_mode=None):
        self.env = gym.make("LunarLander-v2", render_mode=render_mode)  # 当render_mode为"human"时，环境会在每次调用step()方法时自动渲染。因此，在这个特定的情况下，不需要显式调用env.render()。
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

            self.render()

    def render(self):
        """
         render() 函数不是用于弹出窗口实时渲染可视化的；如果以渲染为目的，只需要设置为human就好了；调用render()是为了获取特定格式的返回数组
        :return:
        """
        return self.env.render()

    def close(self):
        self.env.close()


if __name__ == "__main__":
    simulator = LunarLanderSimulator(render_mode="human")
    simulator.print_env_info()
    simulator.run_simulation()
    simulator.close()
