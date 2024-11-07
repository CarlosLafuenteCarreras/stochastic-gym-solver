import numpy as np
import argparse
import gymnasium as gym
import tensorboardX
from git.repo import Repo
import torch
import tqdm

from common import get_file_descriptor, splash_screen
from episode_runner import run_simulation
from models.nn_model import NeuralNetworkModel
from solver.nes_demo import NES, sample_distribution


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
    params.hidden_layers = [8, 8] # [64, 64]

    params.batch_size = 5
    params.repetitions = 5
    params.max_steps = 130

    params.episodes = 50_000 # 10000

    # hiperparameters
    params.learning_rate = 0.1
    params.sigma = 0.5 # 0.01
    params.npop = 50 # 50

    w = NeuralNetworkModel(params.input_size, params.output_size, params.hidden_layers)

    if params.resume:
        w.load_state_dict(torch.load(params.resume))

    population = [w.new_from_parameters(w.get_parameters()) for _ in range(params.npop)]

    logger = splash_screen(params)

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


    episodes = tqdm.trange(params.episodes)
    
    for i in episodes:
        w_tries_numpy = sample_distribution(w, population, params.sigma, params.npop)

        fitness = fitness_function(population, i)
        
        theta = NES(w_tries_numpy, fitness, params.learning_rate, w.get_parameters(), params.npop, params.sigma)
        w.set_parameters(theta)

        if i % 10 == 0:
            reference_fitness, _ = run_simulation([w], # type: ignore
                                        params.env, 
                                        params.max_steps, 
                                        repetitions=100, 
                                        batch_size=10,
                                        progress_bar=False,
                                    )
            
            episodes.set_description(f"Fitness: {reference_fitness.mean():.2f}")
            logger.add_scalar("reference_fitness", reference_fitness.mean(), i)

            parameters = w.get_parameters()

            logger.add_histogram("w_params", parameters, i)

        if i % 100 == 0:
            # save w to disk
            descrp = get_file_descriptor(params, i)

            torch.save(w.state_dict(), descrp)

        if i > 300:
            params.max_steps = 180

        logger.flush()
        

    pass


if __name__ == '__main__':
    run()