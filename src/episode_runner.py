import gymnasium as gym
from tqdm import tqdm

from models import Model

def run_simulation(model: Model, 
                   env: gym.Env, 
                   max_steps: int, 
                   repetitions: int = 1, 
                   render: bool = False, 
                   debug: bool = False, 
                   show_observation: bool = False, 
                   show_action: bool = False):
    def run_once():    
        observation, info = env.reset()
        fitness = 0
        iterator = range(max_steps)
        for _ in range(max_steps):
            action = model.make_decision(observation)
            observation, reward, terminated, truncated, info = env.step(action)

            if show_observation:
                print(f"Observation: {observation}")
            if show_action:
                print(f"Action: {action}")

            fitness += reward

            if terminated or truncated:
                break



    env.close()




