import numpy as np
from scipy.stats import norm

# Expected Improvement
def expected_improvement(x, trainLabels, model, best_y, factor = 2.0):
    """ Functionality of the Expected Improvement acquisition function.
       This function estimates how much will a data point improve a model if it is added next to the training dataset.
       Parameters:
       x:           The set of datapoints that is available for selection.
       trainLabels: The set of features intended to be used for training.
       model:       The model to be trained (usually a Gaussian Process)
       best_y:      The best target value found so far. (Could be minimum or maximum depending on the problem.)
       factor:      A factor used as a weight on the exploration-exploitation tradeoff."""

    x_train = x[trainLabels].to_numpy()

    y_pred, y_std = model.predict(x_train, return_std=True)
    # y_std  = y_pred.std()


    # Maximization Problem
    # z = np.divide(np.subtract(y_pred, best_y + factor), y_std)
    # ei = (np.subtract(y_pred, best_y + factor) * norm.cdf(z)) + (y_std * norm.pdf(z))

    # Minimization Problem
    z = np.divide(np.subtract(best_y - factor, y_pred), y_std)
    ei = (np.subtract(best_y - factor, y_pred + factor) * norm.cdf(z)) + (y_std * norm.pdf(z))


    # Expeted Improvement Considering the mean score of a ZIF for all gasses.
    x["expectedImprovement"] = ei.tolist()
    allZIFs = x.type.unique()

    # meanExpectedImprovement = {}
    # for zif in allZIFs:
    #     meanExpectedImprovement[zif] = x[x['type'] == zif].expectedImprovement.sum() / len(x[x["type"] == zif])

    #     # Expected Improvement Considering The Sum of Scores for All Gasses
    #     # meanExpectedImprovement[zif] = x[x['type'] == zif].expectedImprovement.sum()

    # return ei, max(meanExpectedImprovement, key=meanExpectedImprovement.get)

    # Expected Improvement Considering the gass as a zif feature
    return ei, x.iloc[np.argmax(ei)]['type']