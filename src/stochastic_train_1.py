import torch
import torch.nn as nn
import numpy as np
import argparse
import gymnasium as gym
import tensorboard
from git.repo import Repo

from common import splash_screen
from models.nn_model import NeuralNetworkModel

repo = Repo(search_parent_directories=True)

parser = argparse.ArgumentParser(description='Stochastic Neural Network')

parser.add_argument("--resume", type=str, default=None, help="resume from a model") 

args = parser.parse_args()

params = argparse.Namespace()

params.__dict__.update(args.__dict__)
params.env = ("LunarLander-v3", dict(continuous=True))
params.version = "v1"
params.commit = repo.head.commit.hexsha

env = gym.make(params.env[0], **params.env[1])

params.input_size = env.observation_space.shape[0] # type: ignore
params.output_size = env.action_space.shape[0] # type: ignore
params.hidden_layers = [64, 64]

# hiperparameters
params.learning_rate = 0.01

network = NeuralNetworkModel(params.input_size, params.output_size, params.hidden_layers)

logger = splash_screen(params)

# logger.add_hparams(
#     hparam_dict={
#         "learning_rate": params.learning_rate,
#     },
#     metric_dict={
#         "fitness": 0.0,
#         "max_fitness": 0.0,
#         "steps": 0,
#     },
# )

logger.flush()

def run():
    pass





if __name__ == '__main__':
    run()