from .nn import NeuralNetworkFactory
import inspect

def get_all_methods():
    factory = NeuralNetworkFactory()

    neural_network_methods = {
        name: method for name, method in inspect.getmembers(factory, predicate=inspect.ismethod)
        if name.startswith("GetNN")
    }
    
    return neural_network_methods