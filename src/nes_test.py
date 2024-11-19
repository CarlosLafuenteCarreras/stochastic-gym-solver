import numpy as np
from solver.nes_demo import NES, sample_distribution, sample_distribution_numpy

# Define a simple polynomial function to minimize
def polynomial_function(x):
    return (x - 3) ** 2 + 2

# Define a simple fitness function
def fitness_function(params, i):
    fitness = np.array([polynomial_function(param) for param in params])
    fitness = fitness.mean(axis=1)
    fitness = -fitness
    # add noise to the fitness
    fitness += np.random.randn(fitness.shape[0]) * 0.5
    return fitness

# Initialize parameters
param_size = 1
npop = 10
sigma = 0.1
learning_rate = 0.025
episodes = 1000

# Create a simple array of parameters
theta = np.random.randn(param_size)
population = np.array([theta + sigma * np.random.randn(param_size) for _ in range(npop)])

# Run NES algorithm
for i in range(episodes):
    w_tries_numpy = sample_distribution_numpy(theta, population, sigma, npop)
    fitness = fitness_function(w_tries_numpy, i)
    
    # Debugging information
    print(f"Episode {i}, Fitness: {fitness.mean()}")
    print(f"w_tries_numpy shape: {w_tries_numpy.shape}")
    print(f"fitness shape: {fitness.shape}")
    print(f"theta shape: {theta.shape}")
    
    theta, delta = NES(w_tries_numpy, fitness, learning_rate, theta, npop, sigma)

    if i % 10 == 0:
        print(f"Episode {i}, Fitness: {fitness.mean()}")

print("Final parameters:", theta)