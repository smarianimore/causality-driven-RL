import importlib

import networkx as nx
import yaml
from causalnex.structure import StructureModel
from matplotlib import pyplot as plt

" ******************************************************************************************************************** "


class ConfiguratorParser:
    def __init__(self, config_file):
        with open(config_file, 'r') as file:
            self.config = yaml.safe_load(file)

    def get_environment_config(self):
        return self.config['environment']

    def get_algorithm_config(self):
        return self.config['algorithm']

    def get_simulation_config(self):
        return self.config['simulation']


" ******************************************************************************************************************** "


def dynamic_import(module_name, class_name):
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


" ******************************************************************************************************************** "
