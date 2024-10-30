import itertools
import gymnasium as gym
from joblib import delayed
from tqdm import tqdm
import copy
from concurrent.futures import ThreadPoolExecutor
from models import Model


def run_simulation(models: list[Model], 
                   env: gym.Env, 
                   max_steps: int, 
                   repetitions: int = 1, 
                   render: bool = False, 
                   debug: bool = False, 
                   show_observation: bool = False, 
                   show_action: bool = False):
    def run_once(model: Model, env: gym.Env, max_steps: int, render: bool, debug: bool, show_observation: bool, show_action: bool):    
        observation, _ = env.reset()
        fitness = 0.0
        iterator = range(max_steps)
        for _ in iterator:
            action = model.make_decision(observation)
            observation, reward, terminated, truncated, _ = env.step(action)

            if show_observation:
                print(f"Observation: {observation}")
            if show_action:
                print(f"Action: {action}")

            fitness += float(reward)

            if terminated or truncated:
                break
        
        return fitness

    if repetitions == 1:
        return run_once(models[0], env, max_steps, render, debug, show_observation, show_action)


    tasks = [
        (model, copy.deepcopy(env), max_steps, False, False, False, False)
        for model in itertools.islice(itertools.cycle(models), repetitions*len(models))
    ] 

    # run and collect the results
    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(lambda args: run_once(*args), tasks), total=len(tasks)))

    return results


if __name__ == "__main__":
    env = gym.make("BipedalWalker-v3")

    from models import RandomModel

    models = [RandomModel() for _ in range(50)] # type: list[Model]

    results = run_simulation(models, env, 300, repetitions=10, render=False, debug=False, show_observation=False, show_action=False)

    print(results)

