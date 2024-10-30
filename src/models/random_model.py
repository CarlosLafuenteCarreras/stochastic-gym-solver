


import numpy as np
from models import Model


class RandomModel(Model):   
    def make_decision(self, observation: np.ndarray) -> np.ndarray:
        return np.random.uniform(-1, 1, 4)
    
