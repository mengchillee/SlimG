from models.proposed import OurModel


def load_model(num_nodes, num_features, num_classes, **kwargs):
    return OurModel(num_features, num_nodes, num_classes)