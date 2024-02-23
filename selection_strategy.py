import numpy as np
from abc import ABC
import acquisition_functions as af

class SelectionStrategy(ABC):
    def select_next_instance():
        pass

class GreedySelectionStrategy(SelectionStrategy):
    def __init__():
        pass

    def select_next_instance(self, acquisition_values : np.array, candidate_instances: list):
        return candidate_instances[np.argmax(acquisition_values)]
                
class RandomSelectionStrategy(SelectionStrategy):
    def select_next_instance(self, acquisition_values : np.array, candidate_instances: list):
        raise NotImplementedError("Vassili kanto!" )

class ProbabilisticSelectionStrategy(SelectionStrategy):
    def select_next_instance(self, acquisition_values : np.array, candidate_instances: list):
        raise NotImplementedError("Vassili kanto!")
