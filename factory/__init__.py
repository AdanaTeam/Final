from .nn import NeuralNetworkFactory
import inspect

factory = NeuralNetworkFactory()

neural_network_methods = {
    name: method for name, method in inspect.getmembers(factory, predicate=inspect.ismethod)
    if name.startswith("GetNN")
}

def get_neural_network(method_name):
    if method_name in neural_network_methods:
        return neural_network_methods[method_name]()
    else:
        raise ValueError(f"Метод {method_name} не найден.")
