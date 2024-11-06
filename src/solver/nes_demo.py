import numpy as np

from src.models import NeuralNetworkModel


# from models.nn_model import NeuralNetworkModel
# np.random.seed(0)

# # the function we want to optimize
# def f(w):
#   reward = -np.sum(np.square(solution - w))
#   return reward

# # hyperparameters
# npop = 50 # population size
# sigma = 0.1 # noise standard deviation
# alpha = 0.001 # learning rate

# solution = np.array([0.5, 0.1, -0.3])
# w = np.random.randn(3)
# for i in range(300):
#   if i % 20 == 0:
#     print('iter %d. w: %s, solution: %s, reward: %f' % 
#           (i, str(w), str(solution), f(w)))

#   N = np.random.randn(npop, 3)
#   R = np.zeros(npop)
#   for j in range(npop):
#     w_try = w + sigma*N[j]
#     R[j] = f(w_try)

#   # Should be faster than for loop
#   R2 = np.array([f(w + sigma *N_i) for N_i in N])

#   A = (R - np.mean(R)) / np.std(R)
#   w = w + alpha/(npop*sigma) * np.dot(N.T, A)


## Dareks idea

def p(size: tuple, sigma: float, params: np.ndarray):
  """
  Generates samples from distribution

  :param size: tuple representing the size of the samples
  :return:
  """
  return params + np.random.randn(*size)*sigma

def gradient():
  raise NotImplementedError

def F_inverse():
  raise NotImplementedError

def params_to_model():
  """
  TODO: implement: get list of model params and return list of models
  :return:
  """
  raise NotImplementedError


def sample_distribution(model: NeuralNetworkModel, sigma: float, npop: int) -> list[NeuralNetworkModel]:
  w = model.get_parameters()

  return [model.new_from_parameters(p(w.shape, sigma, w)) for _ in range(npop)]


def NES(samples:np.ndarray, fitness:np.ndarray, learning_rate:float, theta: np.ndarray, npop:int, sigma:float) -> np.ndarray:
  # samples = p(solution)  # TODO: add distribution parameters
  # models = params_to_model(samples)

  # fitness = np.array([f(sample) for sample in samples])

  alpha = learning_rate / (npop * sigma)
  F_inverse = samples.T
  gradient = (fitness - np.mean(fitness)) / np.std(fitness)

  d_theta = alpha * F_inverse @ gradient
  theta += d_theta

  return theta


  #  A = (R - np.mean(R)) / np.std(R)
  #  w = w + alpha / (npop * sigma) * np.dot(N.T, A)



