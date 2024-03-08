# Surrogate model imports
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import ConstantKernel
from sklearn.model_selection import KFold

# Acquisition functions
from acquisition_functions import ExpectedImprovementCalculator
from selection_strategy    import GreedySelectionStrategy

# Metric imports
from sklearn import metrics
import math

# Data handling imports
import numpy as np
import pandas as pd

# Logging
import logging

from sklearn.model_selection import LeavePOut

from abc import ABC

class OptimizationFactory(ABC):
    def optimizeModel():
        pass

class BayesianOptimization(OptimizationFactory):

    def __init__(self):
        pass

    def optimizeModel(self, model : any, zifs : pd.DataFrame, X_featureNames : list, Y_featureNames : list, logger : logging.Logger) -> pd.DataFrame:

        """ Bayesian Optimization As A Method For Optimizing MAE of LogD 
            model:              The model to be optimized.
            zifs :              The data used during optimization.
            X_featureNames:     The names of the training features.
            Y_featureNames:     The names of the target features.
        """

        # Make a list with all unique ZIF names.
        uniqueZIFs = zifs.type.unique()

        # Initiate a gaussian process model
        gp_model = GaussianProcessRegressor(kernel=ConstantKernel(1.0) * RBF(1.0))


        # Initialize dictionary of errors per training data size
        maePerTrainSize = {}
        for leaveOutZifIndex in range(len(uniqueZIFs)):
            
            logger.info("----------   Round " + str(leaveOutZifIndex + 1) + "     ----------")

            trainZIFnames = np.delete(uniqueZIFs, leaveOutZifIndex)
            testZIFname   = uniqueZIFs[leaveOutZifIndex]

            trainZIFs = zifs[zifs['type'] != testZIFname]
            testZIFs  = zifs[zifs['type'] == testZIFname]

            selectRandomSample = 0
            currentData   = pd.DataFrame()
            currentBayesianMae = []
            for sizeOfTrainZIFs in range(len(uniqueZIFs) - 1):

                if selectRandomSample < 5:
                    # Sample 5 random ZIFs.
                    startingZIF  = np.random.choice(trainZIFnames, size=1, replace=False)[0]
                    selectedZIF  = trainZIFs[(trainZIFs['type'] == startingZIF)]

                    # Remove the sellected ZIF from the list of available for training
                    trainZIFs     = trainZIFs[(trainZIFs['type']) != startingZIF]
                    trainZIFnames = np.delete(trainZIFnames, np.where(trainZIFnames == startingZIF))

                    selectRandomSample += 1
                else:
                    # Calculate the expected improvement values for all candidate zifs
                    eiCalculator = ExpectedImprovementCalculator(factor=0)
                    eI = eiCalculator.get_acquisition_function(trainZIFs, X_featureNames, gp_model, minMae)

                    # Select the next zif in a greedy manner
                    greedySelection = GreedySelectionStrategy()
                    eiName = greedySelection.select_next_instance(eI, trainZIFs)
                    selectedZIF = trainZIFs[(trainZIFs['type'] == eiName)]

                    # Remove the sellected ZIF from the list of available for training
                    trainZIFs = trainZIFs[(trainZIFs['type']) != eiName]
                    trainZIFnames = np.delete(trainZIFnames, np.where(trainZIFnames == eiName))

                # Add the next ZIF to the currently used data.
                currentData = pd.concat([currentData, selectedZIF], axis=0, ignore_index=True)

                # Create feature matrices for all currently used data.
                x_trainAll = currentData[X_featureNames].to_numpy()
                y_trainAll = currentData[Y_featureNames].to_numpy()

                # Leave One Out for Bayesian Optimization
                currentBatchNames = currentData.type.unique()
                trainLength = len(currentBatchNames)
                averageMAE = 100.0  # Temporary value denoting that train size 1 has a very large error.
                minMae     = float('-inf')

                # Trying KFold Method
                if trainLength >= 5:
                    averageMAE = 0
                    
                    leaveOutNum = None
                    if trainLength < 10:
                        leaveOutNum = trainLength
                    else:
                        leaveOutNum = 10

                    kf = KFold(n_splits=leaveOutNum)
                    for train_index, test_index in kf.split(currentBatchNames):
                        # trainZifNames  = currentBatchNames[train_index]
                        testZifNames   = currentBatchNames[test_index].tolist()

                        trainBatchZIFs = zifs[~zifs['type'].isin(testZifNames)]
                        testBatchZIF   = zifs[zifs['type'].isin(testZifNames)]

                        x_batchTrain   = trainBatchZIFs[X_featureNames].to_numpy()
                        y_batchTrain   = trainBatchZIFs[Y_featureNames].to_numpy()

                        x_batchTest    = testBatchZIF[X_featureNames].to_numpy()
                        y_batchTest    = testBatchZIF[Y_featureNames].to_numpy()

                        model.fit(x_batchTrain, y_batchTrain.ravel())

                        y_batchPred = model.predict(x_batchTest)

                        averageMAE += metrics.mean_absolute_error(y_batchTest,y_batchPred)

                    averageMAE /= trainLength
                    
                    minMae = min(currentBayesianMae)

                for i in range(selectedZIF.shape[0]):
                    currentBayesianMae.append(averageMAE)

                if trainLength >= 5:
                    # Fit the Gaussian process model to the sampled points
                    gp_model.fit(x_trainAll, np.array(currentBayesianMae))            

                # Prediction on outer leave one out test data

                x_test  = testZIFs[X_featureNames].to_numpy()
                y_test  = testZIFs[Y_featureNames].to_numpy()

                model.fit(x_trainAll, y_trainAll.ravel())

                y_pred  = model.predict(x_test)

                # best_y = np.max(y_pred)
                # best_y = np.min(y_pred)

                mae = metrics.mean_absolute_error(y_test, y_pred)

                if (sizeOfTrainZIFs + 1) not in maePerTrainSize.keys():
                    maePerTrainSize[(sizeOfTrainZIFs + 1)] = []
                
                # Append mae to the corresponding dictionary list
                maePerTrainSize[(sizeOfTrainZIFs + 1)].append(mae)

                logger.info("Number of ZIFs in Dataset: " + str((sizeOfTrainZIFs + 1)))
                logger.info("Mean Absolute Error: " + str(mae))


        result_df = pd.DataFrame()
        result_df["sizeOfTrainingSet"] = np.array([iCnt for iCnt in sorted(maePerTrainSize.keys()) ])
        result_df["averageError"] = [ np.array(maePerTrainSize[iCnt]).mean() for iCnt in maePerTrainSize.keys() ]
        result_df["stdErrorOfMeanError"] = [ np.array(maePerTrainSize[iCnt]).std() / math.sqrt(iCnt) for iCnt in maePerTrainSize.keys() ]

        return result_df