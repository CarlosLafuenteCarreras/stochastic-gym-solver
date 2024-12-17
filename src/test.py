import gymnasium as gym
import numpy as np
import argparse
import torch
from importlib.metadata import version
from models.nn_model import NeuralNetworkModel
#['__annotations__', '__class__', '__class_getitem__', '__delattr__', '__dict__', '__dir__', '__doc__', '__enter__', '__eq__', '__exit__', '__format__', '__ge__', '__getattribute__', '__getstate__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__orig_bases__', '__parameters__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__setstate__', '__sizeof__', '__slots__', '__str__', '__subclasshook__', '__weakref__', '_clean_particles', '_create_particle', '_destroy', '_ezpickle_args', '_ezpickle_kwargs', '_is_protocol', '_np_random', '_np_random_seed', 'action_space', 'clock', 'close', 'continuous', 'crash_penalty', 'enable_wind', 'get_wrapper_attr', 'gravity', 'has_wrapper_attr', 'isopen', 'lander', 'metadata', 'moon', 'np_random', 'np_random_seed', 'observation_space', 'particles', 'prev_reward', 'render', 'render_mode', 'reset', 'reward_shaping', 'screen', 'set_wrapper_attr', 'spec', 'step', 'turbulence_power', 'unwrapped', 'wind_power', 'world']

def make_env():
    instance = gym.make("LunarLander-v3", continuous=False, render_mode="human")

    return instance

env = make_env()
               
observation, info = env.reset(seed=42)

parser = argparse.ArgumentParser(description='Stochastic Neural Network')

parser.add_argument("--resume", type=str, default=None, help="model")

args = parser.parse_args()
# TODO: add save_as_csv as parse argument

model_path = args.resume

model = torch.load(model_path, weights_only=False)

# set duration of runtime loop
N=10e6

# stuff for saving observations and actions in csv
save_as_csv = True
if save_as_csv:
    csv_data = ""
    N = 150 * 100

# loop
for _ in range(N):
   observation = torch.tensor(observation, dtype=torch.float32)
   action = model(observation)


   action = np.argmax(action.detach().numpy())
   if save_as_csv:
        csv_data += ','.join([str(round(float(i),2)) for i in observation.to('cpu').detach().numpy()]) + "," + str(action) + "\n"

   #print(action, '\t'.join([str(round(float(i),2)) for i in observation.to('cpu').detach().numpy()]))
   observation, reward, terminated, truncated, info = env.step(action)

   #print(reward)

   if terminated or truncated:
      observation, info = env.reset()

# saving csv
if save_as_csv:
    with open("retain/decision_data" + model_path[7:-4] + ".csv", "w") as file:
        file.write(csv_data)

env.close()