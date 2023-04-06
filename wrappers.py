import gymnasium as gym
from gymnasium.wrappers import RescaleAction, NormalizeObservation, TimeLimit

# Create the base environment
base_env = gym.make("Hopper-v4")

# Wrap the environment with RescaleAction
wrapped_env = RescaleAction(base_env, min_action=0, max_action=1)

# Wrap the environment with NormalizeObservation
wrapped_env = NormalizeObservation(wrapped_env)

# Wrap the environment with TimeLimit
wrapped_env = TimeLimit(wrapped_env, max_episode_steps=200)

# Use the wrapped environment
observation = wrapped_env.reset()
done = False

while not done:
    action = wrapped_env.action_space.sample()
    observation, reward, done, info = wrapped_env.step(action)

wrapped_env.close()
