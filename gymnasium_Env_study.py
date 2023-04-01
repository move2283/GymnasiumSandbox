import gymnasium as gym

# 创建一个"LunarLander-v2"环境
env = gym.make("LunarLander-v2", render_mode="human")

# 了解Env类的核心方法和属性
print("Action space:", env.action_space)
print("Observation space:", env.observation_space)
print("Reward range:", env.reward_range)
print("Environment spec:", env.spec)
print("Environment metadata:", env.metadata)

# 设置随机数生成器种子
env.action_space.seed(123)

# 重置环境
observation, info = env.reset(seed=42)

# 循环执行1000步操作
for _ in range(1000):
    # 随机选择动作（在这里插入你的策略）
    action = env.action_space.sample()

    # 执行动作，获取新的观察、奖励和环境状态
    observation, reward, terminated, truncated, info = env.step(action)

    # 如果环境终止或截断，重置环境
    if terminated or truncated:
        observation, info = env.reset()

# 关闭环境
env.close()
