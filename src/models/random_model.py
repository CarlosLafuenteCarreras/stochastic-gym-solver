import numba
import numpy as np
from .base import Model

class RandomModel(Model): 
    def make_decision(self, observation: np.ndarray) -> np.ndarray:
        return np.random.randint(0, 4, size=())
    
