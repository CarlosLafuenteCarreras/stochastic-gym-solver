import gymnasium as gym
import numpy as np
import argparse
import torch

from models.nn_model import NeuralNetworkModel


env = gym.make("LunarLander-v3", 
               continuous=False, 
               #human display
               render_mode="human")
               
observation, info = env.reset(seed=42)

parser = argparse.ArgumentParser(description='Stochastic Neural Network')

parser.add_argument("--resume", type=str, default=None, help="model")

args = parser.parse_args()


model_path = args.resume

# Ensure model is correctly instantiated
model = NeuralNetworkModel(8, 4, [32, 16])
if model_path:
    model.load_state_dict(torch.load(model_path))

for _ in range(1000):
   observation = torch.tensor(observation, dtype=torch.float32)
   action = model(observation)
   print(action)
   action = np.argmax(action.detach().numpy())
   observation, reward, terminated, truncated, info = env.step(action)

   print(reward)

   if terminated or truncated:
      observation, info = env.reset()
env.close()