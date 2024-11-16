import gymnasium as gym
import numpy as np
import argparse
import torch

from models.nn_model import NeuralNetworkModel


env = gym.make("LunarLander-v3", 
               continuous=True, 
               #human display
               render_mode="human")
               
observation, info = env.reset(seed=42)

parser = argparse.ArgumentParser(description='Stochastic Neural Network')

parser.add_argument("--resume", type=str, default=None, help="model")

args = parser.parse_args()


model_path = args.resume


if model_path:
      model = torch.load(model_path)
else:
      model = NeuralNetworkModel(8, 2, [32, 16])



for _ in range(1000):
   action = model(observation)
   observation, reward, terminated, truncated, info = env.step(action)

   print(reward)

   if terminated or truncated:
      observation, info = env.reset()
env.close()