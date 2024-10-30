import itertools
import gymnasium as gym
from tqdm import tqdm
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from models import Model

def run_once(model: Model, env: gym.Env, max_steps: int, show_observation: bool, show_action: bool):    
    observation, _ = env.reset()
    fitness = 0.0

    for i in range(max_steps):
        action = model.make_decision(observation)
        observation, reward, terminated, truncated, _ = env.step(action)

        if show_observation:
            print(f"Observation: {observation}")
        if show_action:
            print(f"Action: {action}")

        fitness += float(reward)

        if terminated or truncated:
            return fitness, i+1
    return fitness, max_steps


def run_once_thin(model: Model, env: gym.Env, max_steps: int):    
    observation, _ = env.reset()
    fitness = 0.0

    for i in range(max_steps):
        observation, reward, terminated, truncated, _ = env.step(model.make_decision(observation))

        fitness += reward # type: ignore

        if terminated or truncated:
            return fitness, i+1
        
    return fitness, max_steps

def run_once_thin_wrapper(args):
    return run_once_thin(*args)

def build_task_params(args):
    return (args[0], gym.make(args[1]), args[2])

def run_simulation(models: list[Model], 
                   env: str, 
                   max_steps: int, 
                   repetitions: int = 1, 
                   render: bool = False, 
                   show_observation: bool = False, 
                   show_action: bool = False) -> tuple[np.ndarray, np.ndarray]:
    if repetitions == 1:
        render_mode = "human" if render else None
        fitness, lenght = run_once(models[0], gym.make(env, render_mode=render_mode), max_steps, show_observation, show_action)

        return np.array([fitness]), np.array([lenght])

    tasks = [
        (model, gym.make(env), max_steps)
        for model in itertools.islice(itertools.cycle(models), repetitions*len(models))
    ]

    with ProcessPoolExecutor() as executor:
        results = tqdm(executor.map(run_once_thin_wrapper, tasks), total=repetitions*len(models))
        fitnesses, lengths = zip(*results)
        return np.array(fitnesses).reshape(repetitions, len(models)), np.array(lengths).reshape(repetitions, len(models))


if __name__ == "__main__":
    from models import RandomModel
    import time
    models = [RandomModel() for _ in range(500)] # type: list[Model]

    start_time = time.time()
    fitness, lenghts = run_simulation(models, "BipedalWalker-v3", 500, repetitions=10, render=False, show_observation=False, show_action=False)
    end_time = time.time()

    print(f"Execution time: {end_time - start_time} seconds")

    print(fitness, lenghts)
    print(fitness.shape, lenghts.shape)

