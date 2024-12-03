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
    # turn off wind, make it easier
    params.env = ("LunarLander-v3", dict(wind_power=0.1))
    params.version = "v1"
    params.commit = repo.head.commit.hexsha

    
    def make_env():
        instance = gym.make(params.env[0], **params.env[1])

        # reduce the penalty for crashing
        instance.unwrapped.crash_penalty = -100 # type: ignore
        # increase the reward for landing
        instance.unwrapped.landing_reward = 250 # type: ignore
        # # gravity is weaker
        #instance.unwrapped.gravity = -10 # type: ignore
        # wind is weaker
        #instance.unwrapped.wind_power = 16.0 # type: ignore
        
        return instance

    env = make_env()

    params.input_size = env.observation_space.shape[0] # type: ignore
    params.output_size = env.action_space.shape[0] if isinstance(env.action_space, gym.spaces.Box) else env.action_space.n # type: ignore
    params.hidden_layers = [16, 4]
    params.model_penalty = 0.01

    params.eposode_start = 0
    params.batch_size = 10
    params.repetitions = 100
    params.max_steps = 200

    params.episodes = 50_000

    # hiperparameters
    params.step_randomness_to_w_small = 100000
    params.step_randomness_to_w_big = 2000
    params.sigma_random_small = 0.0
    params.sigma_random_big = 0.001
    params.learning_rate = 0.1
    params.sigma = 0.25
    params.npop = 30


    w = NeuralNetworkModel(params.input_size, params.output_size, params.hidden_layers)
    print(w.get_parameters().shape)
    if params.resume:
        w = torch.load(params.resume)
        params.eposode_start = int(params.resume.split("_")[-1].split(".")[0])
        print(f"Resuming from episode {params.eposode_start}")


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
                                        make_env=make_env,
                                        )
                                        
        model_penaties = np.array([model.get_model_penalty()*params.model_penalty for model in models])
        fitness -= model_penaties
            
        if i % 10 == 0:
            logger.add_histogram("fitness_hist", fitness, i)
            logger.add_histogram("model_penalties", model_penaties, i)

        logger.add_scalar("fitness_mean", fitness.mean(), i)
        logger.add_scalar("steps_mean", lenghts.mean(), i)

        return fitness.mean(axis=0)


    episodes = tqdm.trange(
        params.eposode_start, 
        params.episodes + params.eposode_start,
        desc="Fitness",
    )
    
    for i in episodes:
        w_tries_numpy = sample_distribution(w, population, params.sigma, params.npop)

        fitness = fitness_function(population, i)

        theta, delta = NES(w_tries_numpy, fitness, params.learning_rate, w.get_parameters(), params.npop, params.sigma)


        if i % params.step_randomness_to_w_big == 0 and i > 1:
            theta += np.random.normal(loc=0, scale=params.sigma_random_big, size=theta.shape)
        elif i % params.step_randomness_to_w_small == 0 and i > 1:
            theta += np.random.normal(loc=0, scale=params.sigma_random_small, size=theta.shape)

        w.set_parameters(theta)

        if i % 10 == 0:
            reference_fitness, _ = run_simulation([w], # type: ignore
                                        params.env, 
                                        params.max_steps, 
                                        repetitions=200, 
                                        batch_size=10,
                                        progress_bar=False,
                                        make_env=make_env,
                                    )
            
            episodes.set_description(f"Fitness: {reference_fitness.mean():.2f}")
            logger.add_scalar("reference_fitness", reference_fitness.mean(), i)
            logger.add_histogram("w_delta", delta, i)


            parameters = w.get_parameters()

            logger.add_histogram("w_params", parameters, i)

        if i % 100 == 0:
            # save w to disk
            descrp = get_file_descriptor(params, i)

            torch.save(w, descrp)


        params.sigma *= 0.9995

        if params.sigma < 0.05:
            params.sigma = 0.25

        params.learning_rate *= 0.999

        if params.learning_rate < 0.05:
            params.learning_rate = 0.1

        logger.add_scalar("sigma", params.sigma, i)


    pass


if __name__ == '__main__':
    run()