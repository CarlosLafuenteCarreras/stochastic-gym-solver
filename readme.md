# Stochastic Gym Solver 

Stochastic Gym Solver is a project that applies the **Natural Evolution Strategies (NES)** algorithm to solve the **LunarLander-v3** environment from OpenAI's Gymnasium. The project introduces stochastic adaptations, techniques and custom reward shaping to enhance the agent's performance and training time.

![](/resources/small_model/Animation.gif)

## Authors

- **Darek Petersen** [GitHub](https://github.com/BlueCl0wn) 
  - University of Hamburg  
- **Juan Rubio**  
  - University of Zaragoza  
- **Carlos Lafuente**  [Github](https://github.com/juanchinunizar)
  - University of Zaragoza  
- **Maciej Złotorowicz**  
  - Akademia Górniczo-Hutnicza  [GitHub](https://github.com/Lord225) 

![](/resources/videos/Lander-Sample.mp4)

## Features

- **NES Algorithm**: Implements Natural Evolution Strategies for optimization.
- **Easy run traking**
  - Tracks parameters for easy reproduction.
- **Neural Network-based Policy**:
  - Configurable input size, hidden layers, and output size.
- **Logging and Visualization**:
  - Integration with TensorBoard for real-time monitoring.
- **Stochastic Adjustments**:
  - Randomness injections for robust policy exploration.
- **Custom Reward Shaping**:
    - Additional rewards for better policy learning.
- **Multithreading**:
  - Parallelized fitness evaluations for fast training.
- **Hyperparameter Tuning**:
    - Configurable hyperparameters for optimal performance.

## Installation
### Create venv/env with `python=3.10.15`
```
conda env create -f environment.yml 
pip install -r requirements.txt
```

## Usage
### Train script
```
python stochastic_train_1.py
python stochastic_train_1.py --resume <path_to_checkpoint>
```

tensorboard logs are stored in `./logs` directory

### Test script
```
python test.py --resume <path_to_checkpoint>
```

## Natural Gradient
Natural Evolution Strategies (NES) is a novel approach to optimize unknown fitness func-
tions within a black-box optimization framework. It allows us to search for good or near-
optimal solutions to numerous difficult real-world. problems.
The key characteristics of NES include:
* It maintains a multinormal distribution and iteratively updates over the candidate
solution space.
* Updates are performed using the Natural Gradient to improve expected fitness.
* Incorporates innovations such as optimal fitness baselines and importance mixing
to reduce fitness evaluations.

NES algorithm has tendecy to converge to a local minimum, which is why we have implemented a set of tools to enhance the exploration of the policy space.
Including:
* Randomness injections to the policy
* Varible hiperparameters
* Custom reward shaping to guide the agent to the desired behavior

More on NES algorithm in linked paper.



## Hiperparameters
You can ajust the hiperparameters in the `stochastic_train_1.py` file.

* `params.hidden_layers`
    * Specifies the architecture of the policy network. NES algorithm can handle varius network architectures, but we recommend simple architectures for with few neurons in each layer. (for example `[12, 4]` or `[4]`)

* `params.model_penalty`
    * l2 regularization penalty for the policy network.

* `params.batch_size`
    * Defines the number of episodes in each batch. Reduces amount of instances of environment. Recommened values is `2-10`

* `params.repetitions`
    * Number of fitness evaluations per episode, providing robustness in estimating the policy’s quality. Higher values improve the quality of the policy but increase computational cost. Recommended values are `10-200`.

* `params.max_steps`
    * Max number of steps in each episode.

* `params.episodes`
    * Stopping criterion for the training process.

* `params.step_randomness_to_w_small` and `params.step_randomness_to_w_big`:
    * Frequency of introducing small and large random perturbations to the policy

* `params.sigma_random_small` and `params.sigma_random_big`
    * The magnitude of small and large random adjustments, respectively. These values control how much randomness is injected into the policy at each step. Recommended values are `0.001-0.01` for small and `0.05-0.15` for large perturbations. Magniute is depentent frequency

* `params.learning_rate`
    * Step size used to update the distribution parameters in NES. Higher values speed up training but risk overshooting optimal solutions. Recommended values are `0.01-0.5`.

* `params.sigma`
    The standard deviation of the noise distribution used for sampling perturbations. Higher values increase exploration but may slow down convergence. Recommended values are `0.5-5.0`.

* `params.npop`
    * The population size, representing the number of candidate policies evaluated per generation. Higher values improve robustness but require more computational resources. Recommended values are `10-100`.



## Custom Reward Shaping
The LunarLander-v3 environment is a challenging task that requires the agent to learn a complex policy to land the spacecraft safely. To facilitate the learning process, we have implemented custom reward shaping to guide the agent towards the desired behavior.
