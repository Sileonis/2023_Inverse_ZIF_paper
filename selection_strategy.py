import numpy as np
import pandas as pd
from abc import ABC
import acquisition_functions as af

class SelectionStrategy(ABC):
    def select_next_instance():
        pass

class GreedySelectionStrategy(SelectionStrategy):
    def __init__(self):
        pass

    def select_next_instance(self, acquisition_values : np.array, candidate_instances: pd.DataFrame):
        return candidate_instances.iloc[np.argmax(acquisition_values)]["type"]
                
class RandomSelectionStrategy(SelectionStrategy):
    def select_next_instance(self, acquisition_values : np.array, candidate_instances: pd.DataFrame):
        return np.random.choice(candidate_instances.type.unique(), size=1, replace=False)[0]

class ProbabilisticSelectionStrategy(SelectionStrategy):
    def select_next_instance(self, acquisition_values : np.array, candidate_instances: pd.DataFrame):
        raise NotImplementedError("Vassili kanto!")
