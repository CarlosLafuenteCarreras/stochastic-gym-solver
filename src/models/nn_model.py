import numpy as np
from base import Model
import torch
import torch.nn as nn

class NeuralNetworkModel(Model, nn.Module):
    def __init__(self, input_size: int, hidden_layers: list, output_size: int):
        super(NeuralNetworkModel, self).__init__()

        layers = ([nn.Linear(input_size, hidden_layers[0])] # List of layers.
                  +[nn.Linear(hidden_layers[i], hidden_layers[i+1]) for i in range(len(hidden_layers)-1)]
                  +[nn.Linear(hidden_layers[-1], output_size)])
        act_funcs = [nn.ReLU()] * len(hidden_layers) + [nn.Sigmoid()] # List of activation functions.

        zipped = [elem for pair in zip(layers, act_funcs) for elem in pair] # Creates list with all layers and activation functions.

        self.linear = nn.Sequential(*zipped)


    def make_decision(self, observation: np.ndarray) -> np.ndarray:
        raise NotImplementedError()
    
    def get_parameters(self) -> dict:
        parameters = {}
        for name, param in self.linear.named_parameters():
            parameters[name] = param.detach().numpy()  # Detach and convert to NumPy array
        return parameters

    def get_parametes_Iterator(self):
        return self.linear.parameters() # coutputs Iterator[Parameter]

    def flatten_parameters(self):
        """
        Nimmt ein PyTorch-Modell und gibt alle Parameter als 1-dimensionalen Array zurück.
        """
        parameters = [param.data.view(-1) for param in self.linear.parameters()]
        return torch.cat(parameters)

    def assign_parameters(self, flat_params):
        """
        Nimmt einen 1-dimensionalen Array und überschreibt damit die Parameter im Modell.
        """
        current_position = 0
        for param in self.linear.parameters():
            # Bestimme die Anzahl der Datenpunkte in den Parametern
            param_size = param.numel()
            # Erhalte den entsprechenden Abschnitt des Arrays
            param.data.copy_(flat_params[current_position:current_position + param_size].view_as(param))
            # Update die aktuelle Position
            current_position += param_size
    
    def set_parameters(self, parameters: dict):
        self.linear.
        raise NotImplementedError()


if __name__ == "__main__":
    model = NeuralNetworkModel(2, [3], 1)
    print("model ", model)
    print()
    print("model.linear.params ", list(model.linear.parameters()))#
    print()
    print("model.get_params ", model.get_parameters())

