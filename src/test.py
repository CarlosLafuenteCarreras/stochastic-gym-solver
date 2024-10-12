import gymnasium as gym
import numpy as np

env = gym.make("LunarLander-v3", render_mode="human")
observation, info = env.reset(seed=42)
for _ in range(1000):
   action = env.action_space.sample()
   observation, reward, terminated, truncated, info = env.step(action)

   print(reward)

   if terminated or truncated:
      observation, info = env.reset()
env.close()