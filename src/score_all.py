import gymnasium as gym
import numpy as np
import argparse
import torch
import episode_runner
import tensorboardX
from episode_runner_raw import run_simulation
from models.nn_model import NeuralNetworkModel


def make_env():
    instance = gym.make("LunarLander-v3", continuous=False)
    
    return instance


def get_episode(resume: str):
    return int(resume.split("_")[-1].split(".")[0])

def main():
    TO_SCORE_PATH = "./models"

    import os
    import tqdm

    values = []

    for model in tqdm.tqdm(os.listdir(TO_SCORE_PATH)):
        if model.endswith(".pth"):
            try:
                model_path = os.path.join(TO_SCORE_PATH, model)
                model = torch.load(model_path, weights_only=False)
                env = make_env()
                fitness = run_simulation([model], env, 200, repetitions=200, batch_size=10, progress_bar=False, make_env=make_env)
                fitness = fitness[0].mean()
                values.append((model_path, fitness, get_episode(model_path)))
            except Exception as e:
                print(f"Error with model {model}: {e}")
    import pandas as pd

    df = pd.DataFrame(values, columns=["model", "fitness", "episode"])

    df.to_csv("scores.csv")

if __name__ == "__main__":
    main()