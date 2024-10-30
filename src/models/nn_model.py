import numpy as np
from models import Model


class NeuralNetworkModel(Model):
    def __init__(self, input_size: int, hidden_layers: list):
        pass

    def make_decision(self, observation: np.ndarray) -> np.ndarray:
        raise NotImplementedError()
    
    def get_parameters(self) -> dict:
        raise NotImplementedError()
    
    def set_parameters(self, parameters: dict):
        raise NotImplementedError()
