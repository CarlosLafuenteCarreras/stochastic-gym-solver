import numpy as np

class Model:
    def make_decision(self, observation: np.ndarray) -> np.ndarray:
        raise NotImplementedError()
    
    def get_parameters(self) -> dict:
        raise NotImplementedError()
    
    def set_parameters(self, parameters: dict):
        raise NotImplementedError()

