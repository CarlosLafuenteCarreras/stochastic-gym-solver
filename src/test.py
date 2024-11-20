import gymnasium as gym
import numpy as np
import argparse
import torch

from models.nn_model import NeuralNetworkModel
#['__annotations__', '__class__', '__class_getitem__', '__delattr__', '__dict__', '__dir__', '__doc__', '__enter__', '__eq__', '__exit__', '__format__', '__ge__', '__getattribute__', '__getstate__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__orig_bases__', '__parameters__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__setstate__', '__sizeof__', '__slots__', '__str__', '__subclasshook__', '__weakref__', '_clean_particles', '_create_particle', '_destroy', '_ezpickle_args', '_ezpickle_kwargs', '_is_protocol', '_np_random', '_np_random_seed', 'action_space', 'clock', 'close', 'continuous', 'crash_penalty', 'enable_wind', 'get_wrapper_attr', 'gravity', 'has_wrapper_attr', 'isopen', 'lander', 'metadata', 'moon', 'np_random', 'np_random_seed', 'observation_space', 'particles', 'prev_reward', 'render', 'render_mode', 'reset', 'reward_shaping', 'screen', 'set_wrapper_attr', 'spec', 'step', 'turbulence_power', 'unwrapped', 'wind_power', 'world']

def make_env():
    instance = gym.make("LunarLander-v3", continuous=False, render_mode="human")

    instance.unwrapped.reward_shaping = True # type: ignore
    # reduce the penalty for crashing
    instance.unwrapped.crash_penalty = -10 # type: ignore
    # # reduce initial velocity
    # print(dir(instance.unwrapped))
    instance.unwrapped.initial_random = 0.00 # type: ignore
    # # gravity is weaker
    instance.unwrapped.gravity = -3 # type: ignore
    # wind is weaker
    #instance.unwrapped.wind_power = 0 # type: ignore
    
    return instance

env = make_env()
               
observation, info = env.reset(seed=42)

parser = argparse.ArgumentParser(description='Stochastic Neural Network')

parser.add_argument("--resume", type=str, default=None, help="model")

args = parser.parse_args()


model_path = args.resume

model = torch.load(model_path, weights_only=False)

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