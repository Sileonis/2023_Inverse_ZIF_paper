import numpy as np
from scipy.stats import norm
from abc import ABC

class AcquisitionCalculator(ABC):
    def get_acquisition_function(x, featureNames, model, best_y, factor):
        pass
    
class ExpectedImprovementCalculator(AcquisitionCalculator):
   def __init__(self, factor):
      self.factor = factor

   def get_acquisition_function(self, x, featureNames, model, best_y):
      return self.expected_improvement(x, featureNames, model, best_y, self.factor)
    
        
   # Expected Improvement
   def expected_improvement(self, x, featureNames, model, best_y, factor = 2.0):
      """ Functionality of the Expected Improvement acquisition function.
         This function estimates how much will a data point improve a model if it is added next to the training dataset.
         Parameters:
         x:           The dataframe that contains candidate instances to select.
         featureNames: The names of the features contained in dataframe x, which will be used to create the training vectors.
         model:       The trained model which maps instances in the x[featureNames] subspace to an estimate of the target variable y (usually a Gaussian Process)
         best_y:      The best target value y found so far. (Could be minimum or maximum depending on the problem.)
         factor:      A factor used as a weight on the exploration-exploitation tradeoff."""

      x_global = x[featureNames].to_numpy()

      y_pred, y_std = model.predict(x_global, return_std=True)

      # Maximization Problem
      # z = np.divide(np.subtract(y_pred, best_y + factor), y_std)
      # ei = (np.subtract(y_pred, best_y + factor) * norm.cdf(z)) + (y_std * norm.pdf(z))

      # Minimization Problem
      z = np.divide(np.subtract(best_y - factor, y_pred), y_std)
      ei = (np.subtract(best_y - factor, y_pred + factor) * norm.cdf(z)) + (y_std * norm.pdf(z))


      # Expeted Improvement Considering the mean score of a ZIF for all gasses.
      x["expectedImprovement"] = ei.tolist()

      # Expected Improvement Considering the gass as a zif feature
      return ei