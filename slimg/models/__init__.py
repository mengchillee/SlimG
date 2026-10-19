from .proposed import SlimG


def load_model(num_nodes, num_features, num_classes, components=None, **kwargs):
    return SlimG(num_features, num_nodes, num_classes, components=components)