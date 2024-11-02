import numpy as np
import argparse
import gymnasium as gym
import tensorboardX
from git.repo import Repo
import tqdm

from common import splash_screen
from episode_runner import run_simulation
from models.nn_model import NeuralNetworkModel


def run():
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

    params.batch_size = 5
    params.repetitions = 10
    params.max_steps = 150

    params.episodes = 10000

    # hiperparameters
    params.learning_rate = 0.01

    # network = NeuralNetworkModel(params.input_size, params.output_size, params.hidden_layers)

    logger = splash_screen(params)

    # logger.add_hparams(
    #     hparam_dict={
    #         "learning_rate": params.learning_rate,
    #         "hidden_layers": str(params.hidden_layers),
    #     },
    #     metric_dict={
    #         "fitness": 0.0,
    #         "max_fitness": 0.0,
    #         "steps": 0,
    #     },
    # )

    logger.flush()


    def fitness_function(models: list[NeuralNetworkModel], i: int):
        
        fitness, lenghts = run_simulation(models, # type: ignore
                                        params.env, 
                                        params.max_steps, 
                                        repetitions=params.repetitions, 
                                        batch_size=params.batch_size,
                                        progress_bar=False,
                                        )
        
        logger.add_histogram("fitness_hist", fitness, i)
        logger.add_scalar("fitness_mean", fitness.mean(), i)
        logger.add_scalar("max_fitness", fitness.max(), i)
        logger.add_scalar("steps_mean", lenghts.mean(), i)

        return fitness.mean(axis=0)
    
    for i in tqdm.trange(params.episodes):
        fitness = fitness_function([NeuralNetworkModel(params.input_size, params.output_size, params.hidden_layers)], i)
        logger.add_scalar("fitness", fitness, i)
        logger.flush()
        

    pass


if __name__ == '__main__':
    run()