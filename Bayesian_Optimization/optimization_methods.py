import os
import math
import numpy as np
import pandas as pd
import time
import string
import random
from logger import Logger
from abc import ABC
from datetime import timedelta
from sklearn import metrics
from acquisition_functions import ExpectedImprovementCalculator
from selection_strategy    import GreedySelectionStrategy
from selection_strategy    import RandomSelectionStrategy
from selection_strategy    import SerialSelectionStrategy
from selection_strategy    import ProbabilisticSelectionStrategy
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import ConstantKernel
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process.kernels import RationalQuadratic
from sklearn.gaussian_process.kernels import DotProduct
from sklearn.gaussian_process.kernels import WhiteKernel
from sklearn.model_selection import KFold

# TODO Refactor this file by separating functions

class OptimizationFactory(ABC):
    def optimizeModel():
        pass

class BayesianOptimization(OptimizationFactory):

    def __init__(self,logger : Logger):
        self.logger = logger
        self.logPrefix = "Bayesian Optimization"

    def optimizeModel(self, model : any, zifs : pd.DataFrame, X_featureNames : list, Y_featureNames : list , designspace_thres : int, bo_selection : str, save_path : str) -> pd.DataFrame:

        """ Bayesian Optimization As A Method For Optimizing MAE of LogD 
            model:              The model to be optimized.
            zifs :              The data used during optimization.
            X_featureNames:     The names of the training features.
            Y_featureNames:     The names of the target features.
        """
        optimization_start_time = time.time()

        # Initiate a gaussian process model
        #self.logger.info(self.logPrefix, "----------   Using Gaussian Process Regressor as Surrogate with kernel: ConstantKernel(1.0) * RBF(1.0) ----------")
        #gp_model = GaussianProcessRegressor(kernel=ConstantKernel(1.0) * RBF(1.0))
        # self.logger.info(self.logPrefix, "----------   Using Gaussian Process Regressor as Surrogate with kernel: 1.0 * Matern(length_scale=1.0, nu=2.5) ----------")
        # gp_model = GaussianProcessRegressor(kernel=1.0 * Matern(length_scale=1.0, nu=2.5))
        # self.logger.info(self.logPrefix, "----------   Using Gaussian Process Regressor as Surrogate with kernel: RationalQuadratic(length_scale=1.0, alpha=1.5) ----------")
        # gp_model = GaussianProcessRegressor(kernel=RationalQuadratic(length_scale=1.0, alpha=1.5))
        self.logger.info(self.logPrefix, "----------   Using Gaussian Process Regressor as Surrogate with kernel: DotProduct(sigma_0=1.0) + WhiteKernel() ----------")
        gp_model = GaussianProcessRegressor(kernel=DotProduct(sigma_0=1.0) + WhiteKernel())

        # Count the total number that the kfold process takes in seconds
        total_kfold_elapsed_time = 0.0

        # Make a list with all unique ZIF names.
        uniqueZIFs = zifs.type.unique()

        # Initialize dictionary of errors per training data size

        fold_num = 10
        design_space = (len(uniqueZIFs))
        fold_design_space = len(uniqueZIFs) * ((fold_num - 1) / fold_num)

        select_data_points_num = 0
        if design_space < designspace_thres:
            fold_num = len(uniqueZIFs)
            select_data_points_num = len(uniqueZIFs) - 1

        elif fold_design_space < designspace_thres:
            select_data_points_num = int(fold_design_space)

        else:
            select_data_points_num = designspace_thres

        self.logger.info(self.logPrefix,
                         "----------   Data points to be selected are: " + str(select_data_points_num) +  " out of a design space of " + str(design_space) + " data points." + "     ----------")

        zif_kfold = KFold(n_splits=fold_num, shuffle=True, random_state=10)
        inner_round = 0 
        maePerTrainSize = {}
        for train_zif_indicies, left_out_zif_indicies in zif_kfold.split(uniqueZIFs):
            inner_round += 1
            roundPath = os.path.join(save_path, "Round_" + str(inner_round))
            os.mkdir(roundPath)
            roundMae = []

            self.logger.info(self.logPrefix,
                        "----------   Round " + str(inner_round) + "     ----------")

            trainZIFnames = np.delete(uniqueZIFs, left_out_zif_indicies)
            testZIFname   = uniqueZIFs[left_out_zif_indicies]

            trainZIFs = zifs[~zifs['type'].isin(testZIFname)]
            testZIFs  = zifs[zifs['type'].isin(testZIFname)]

            selectRandomSample = 0
            currentData   = pd.DataFrame()
            currentBayesianMae = []

            maeBestPerformanceList      = []
            maeStopCriterionMet         = False
            bestPerformingData          = {}
            
            for sizeOfTrainZIFs in range(select_data_points_num):

                if selectRandomSample < 5:
                    # Sample 5 random ZIFs.
                    randomSelection = RandomSelectionStrategy(logger=self.logger)
                    randomZifName = randomSelection.select_next_instance(trainZIFnames)
                    selectedZIF  = trainZIFs[(trainZIFs['type'] == randomZifName)]

                    # Remove the sellected ZIF from the list of available for training
                    trainZIFs     = trainZIFs[(trainZIFs['type']) != randomZifName]
                    trainZIFnames = np.delete(trainZIFnames, np.where(trainZIFnames == randomZifName))

                    selectRandomSample += 1
                else:
                    # Calculate the expected improvement values for all candidate zifs
                    eiCalculator = ExpectedImprovementCalculator(factor=0,logger = self.logger)
                    eI = eiCalculator.get_acquisition_function(trainZIFs, X_featureNames, gp_model, minMae)

                    eiName = None
                    if bo_selection == "greedy":
                        # Select the next zif in a greedy manner
                        greedySelection = GreedySelectionStrategy(logger=self.logger)
                        eiName = greedySelection.select_next_instance(eI, trainZIFs)
                    elif bo_selection == "prob":
                        # Select the next zif in a probabilistic manner.
                        probSelection   = ProbabilisticSelectionStrategy(logger=self.logger)
                        eiName = probSelection.select_next_instance(eI, trainZIFs)
                    else:
                        raise ValueError("Unknown selection method: " + str(bo_selection))

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
                averageMAE = 100.0  # TODO Temporary value denoting that train size 1 has a very large error.
                minMae     = float('-inf')

                # Trying KFold Method
                if trainLength >= 5:
                    self.logger.info(self.logPrefix,
                                "----- Start of inner KFold method in order to compute the MAE of the model in the currently known zif space. -----")                    

                    averageMAE = 0
                    
                    leaveOutNum = None
                    if trainLength < 10:
                        leaveOutNum = trainLength
                    else:
                        leaveOutNum = 10

                    self.logger.info(self.logPrefix,
                                " --- Starting A " + str(leaveOutNum) + "Fold Process To Compute The Average MAE of the model. --- ")
                    kf = KFold(n_splits=leaveOutNum)
                    kFold_start_time = time.time()
                    for train_index, test_index in kf.split(currentBatchNames):
                        trainZifNames  = currentBatchNames[train_index].tolist()
                        testZifNames   = currentBatchNames[test_index].tolist()

                        trainBatchZIFs = zifs[zifs['type'].isin(trainZifNames)]
                        testBatchZIF   = zifs[zifs['type'].isin(testZifNames)]

                        x_batchTrain   = trainBatchZIFs[X_featureNames].to_numpy()
                        y_batchTrain   = trainBatchZIFs[Y_featureNames].to_numpy()

                        x_batchTest    = testBatchZIF[X_featureNames].to_numpy()
                        y_batchTest    = testBatchZIF[Y_featureNames].to_numpy()

                        model.fit(x_batchTrain, y_batchTrain.ravel())

                        y_batchPred = model.predict(x_batchTest)

                        averageMAE += metrics.mean_absolute_error(y_batchTest,y_batchPred)
                    kFold_elapsed_time = time.time() - kFold_start_time
                    total_kfold_elapsed_time += kFold_elapsed_time
                    self.logger.info(self.logPrefix, 
                                " --- Finished The " + str(leaveOutNum) + "Fold Process In --- " + 
                                time.strftime("%H:%M:%S", time.gmtime(kFold_elapsed_time)) + " time.")
                    
                    averageMAE /= trainLength # TODO: Check if this is correct
                    
                for i in range(selectedZIF.shape[0]):
                    currentBayesianMae.append(averageMAE)

                if currentBayesianMae:
                    minMae = min(currentBayesianMae)

                if trainLength >= 5:
                    # Fit the Gaussian process model to the sampled points
                    gp_model.fit(x_trainAll, np.array(currentBayesianMae))            

                # Prediction on outer leave one out test data

                x_test  = testZIFs[X_featureNames].to_numpy()
                y_test  = testZIFs[Y_featureNames].to_numpy()

                model.fit(x_trainAll, y_trainAll.ravel())

                y_pred  = model.predict(x_test)

                mae = metrics.mean_absolute_error(y_test, y_pred)

                saveCurrentData = False
                stopCriterion   = None
                if not maeStopCriterionMet:
                    if len(maeBestPerformanceList) == 5:
                        stopCriterion = "low_performance_gain"
                        maeStopCriterionMet = True
                    else:

                        if mae <= 0.5:
                            stopCriterion = "error_threshold_reached"
                            maeStopCriterionMet = True
                            saveCurrentData = True
                            bestPerformingData     = {}

                        elif (not maeBestPerformanceList) or (mae > maeBestPerformanceList[0] - (maeBestPerformanceList[0] * 0.1)):
                            maeBestPerformanceList.append(mae)
                            saveCurrentData = True

                        else:
                            maeBestPerformanceList = []
                            bestPerformingData     = {}

                    if saveCurrentData:
                        datasetsInDict = len(bestPerformingData) + 1
                        bestPerformingData[datasetsInDict] = currentData
                        bestPerformingData[datasetsInDict]["mae"] = mae
                        bestPerformingData[datasetsInDict]["tested_against"] = '-'.join(testZIFname)

                    if maeStopCriterionMet:
                        # Save th ecurrent snapshot of the data to a csv file. Use a random name to avoid overwriting
                        for key, value in bestPerformingData.items():
                            bestPerformingData[key]["stop_criterion"] = stopCriterion
                            bestPerformingDataName = "Round_" + str(inner_round) + "_" + "Dataset_No_" + str(key) + "_best_performing_dataset_of_size_" + str(len(value.type.unique())) + "_" + ''.join(random.choice(string.ascii_letters) for _ in range(5)) + ".csv"
                            bestPerformingData[key].to_csv(os.path.join(roundPath,bestPerformingDataName), index=False)


                if (sizeOfTrainZIFs + 1) not in maePerTrainSize.keys():
                    maePerTrainSize[(sizeOfTrainZIFs + 1)] = []
                
                # Append mae to the corresponding dictionary list
                maePerTrainSize[(sizeOfTrainZIFs + 1)].append(mae)

                self.logger.info(self.logPrefix, 
                            "Number of ZIFs in Dataset: " + str((sizeOfTrainZIFs + 1)))
                self.logger.info(self.logPrefix,
                            "Mean Absolute Error: " + str(mae))

                roundMae.append(mae)

            self.logger.info(self.logPrefix,"Writting results of Round " + str(inner_round) + " to file.")

            intermediate_df = pd.DataFrame()
            intermediate_df["sizeOfTrainingSet"]       = list(range(len(roundMae)))
            intermediate_df["averageError"]            = roundMae
            intermediate_df["stdErrorOfMeanError"]     = roundMae
            intermediate_df["stdDeviationOfMeanError"] = roundMae
            intermediate_df.to_csv(os.path.join(roundPath,"full_round_" + str(inner_round) + ".csv"), index=False)

        total_elapsed_time = time.time() - optimization_start_time
        self.logger.info(self.logPrefix,
                    "Execution Time Is: " + str(timedelta(seconds=total_elapsed_time)))
        
        self.logger.info(self.logPrefix,
                    "The kFold process takes " +
                    str(total_kfold_elapsed_time * 100 / total_elapsed_time) +
                    "%% of the total optimization time.")

        result_df = pd.DataFrame()
        result_df["sizeOfTrainingSet"]       = np.array([iCnt for iCnt in sorted(maePerTrainSize.keys()) ])
        result_df["averageError"]            = [ np.array(maePerTrainSize[iCnt]).mean() for iCnt in maePerTrainSize.keys() ]
        result_df["stdErrorOfMeanError"]     = [ np.array(maePerTrainSize[iCnt]).std() / math.sqrt(iCnt) for iCnt in maePerTrainSize.keys() ]
        result_df["stdDeviationOfMeanError"] = [ np.array(maePerTrainSize[iCnt]).std()  for iCnt in maePerTrainSize.keys() ]

        return result_df

class RandomOptimization(OptimizationFactory):

    def __init__(self, logger : Logger):
        self.logger = logger
        self.logPrefix = "Random Optimization"
    
    def optimizeModel(self, model : any, zifs : pd.DataFrame, X_featureNames : list, Y_featureNames : list , designspace_thres : int, bo_selection : str, save_path : str) -> pd.DataFrame:

        """ Random Optimization As A Method For Optimizing MAE of LogD 
            model:              The model to be optimized.
            zifs :              The data used during optimization.
            X_featureNames:     The names of the training features.
            Y_featureNames:     The names of the target features.
        """
        optimization_start_time = time.time()
        # Make a list with all unique ZIF names.
        uniqueZIFs = zifs.type.unique()

        # Count the total number that the kfold process takes in seconds
        total_kfold_elapsed_time = 0.0

        # Initialize dictionary of errors per training data size

        fold_num = 10
        design_space = (len(uniqueZIFs))
        fold_design_space = len(uniqueZIFs) * ((fold_num - 1) / fold_num)

        select_data_points_num = 0
        if design_space < designspace_thres:
            fold_num = len(uniqueZIFs)
            select_data_points_num = len(uniqueZIFs) - 1

        elif fold_design_space < designspace_thres:
            select_data_points_num = int(fold_design_space)

        else:
            select_data_points_num = designspace_thres

        self.logger.info(self.logPrefix,
                         "----------   Data points to be selected are: " + str(select_data_points_num) +  " out of a design space of " + str(design_space) + " data points." + "     ----------")


        zif_kfold = KFold(n_splits=fold_num, shuffle=True, random_state=10)
        inner_round = 0 
        maePerTrainSize = {}
        for train_zif_indicies, left_out_zif_indicies in zif_kfold.split(uniqueZIFs):
            inner_round += 1
            roundPath = os.path.join(save_path, "Round_" + str(inner_round))
            os.mkdir(roundPath)
            roundMae = []

            self.logger.info(self.logPrefix,
                        "----------   Round " + str(inner_round) + "     ----------")

            trainZIFnames = np.delete(uniqueZIFs, left_out_zif_indicies)
            testZIFname   = uniqueZIFs[left_out_zif_indicies]

            trainZIFs = zifs[~zifs['type'].isin(testZIFname)]
            testZIFs  = zifs[zifs['type'].isin(testZIFname)]

            selectRandomSample = 0
            currentData   = pd.DataFrame()

            maeBestPerformanceList      = []
            maeStopCriterionMet         = False
            bestPerformingData          = {}

            for sizeOfTrainZIFs in range(select_data_points_num):

                # Sample each ZIF randomly.
                randomSelection = RandomSelectionStrategy(logger=self.logger)
                randomZifName = randomSelection.select_next_instance(trainZIFnames)
                selectedZIF  = trainZIFs[(trainZIFs['type'] == randomZifName)]

                # Remove the sellected ZIF from the list of available for training
                trainZIFs     = trainZIFs[(trainZIFs['type']) != randomZifName]
                trainZIFnames = np.delete(trainZIFnames, np.where(trainZIFnames == randomZifName))

                selectRandomSample += 1

                # Add the next ZIF to the currently used data.
                currentData = pd.concat([currentData, selectedZIF], axis=0, ignore_index=True)

                # Create feature matrices for all currently used data.
                x_trainAll = currentData[X_featureNames].to_numpy()
                y_trainAll = currentData[Y_featureNames].to_numpy()

                # Prediction on outer leave one out test data
                x_test  = testZIFs[X_featureNames].to_numpy()
                y_test  = testZIFs[Y_featureNames].to_numpy()

                model.fit(x_trainAll, y_trainAll.ravel())

                y_pred  = model.predict(x_test)

                mae = metrics.mean_absolute_error(y_test, y_pred)

                saveCurrentData = False
                stopCriterion   = None
                if not maeStopCriterionMet:
                    if len(maeBestPerformanceList) == 5:
                        stopCriterion = "low_performance_gain"
                        maeStopCriterionMet = True
                    else:

                        if mae <= 0.5:
                            stopCriterion = "error_threshold_reached"
                            maeStopCriterionMet = True
                            saveCurrentData = True
                            bestPerformingData     = {}

                        elif (not maeBestPerformanceList) or (mae > maeBestPerformanceList[0] - (maeBestPerformanceList[0] * 0.1)):
                            maeBestPerformanceList.append(mae)
                            saveCurrentData = True

                        else:
                            maeBestPerformanceList = []
                            bestPerformingData     = {}

                    if saveCurrentData:
                        datasetsInDict = len(bestPerformingData) + 1
                        bestPerformingData[datasetsInDict] = currentData
                        bestPerformingData[datasetsInDict]["mae"] = mae
                        bestPerformingData[datasetsInDict]["tested_against"] = '-'.join(testZIFname)

                    if maeStopCriterionMet:
                        # Save th ecurrent snapshot of the data to a csv file. Use a random name to avoid overwriting
                        for key, value in bestPerformingData.items():
                            bestPerformingData[key]["stop_criterion"] = stopCriterion
                            bestPerformingDataName = "Round_" + str(inner_round) + "_" + "Dataset_No_" + str(key) + "_best_performing_dataset_of_size_" + str(len(value.type.unique())) + "_" + ''.join(random.choice(string.ascii_letters) for _ in range(5)) + ".csv"
                            bestPerformingData[key].to_csv(os.path.join(roundPath,bestPerformingDataName), index=False)


                if (sizeOfTrainZIFs + 1) not in maePerTrainSize.keys():
                    maePerTrainSize[(sizeOfTrainZIFs + 1)] = []
                
                # Append mae to the corresponding dictionary list
                maePerTrainSize[(sizeOfTrainZIFs + 1)].append(mae)

                self.logger.info(self.logPrefix, 
                            "Number of ZIFs in Dataset: " + str((sizeOfTrainZIFs + 1)))
                self.logger.info(self.logPrefix,
                            "Mean Absolute Error: " + str(mae))

                roundMae.append(mae)

            self.logger.info(self.logPrefix,"Writting results of Round " + str(inner_round) + " to file.")

            intermediate_df = pd.DataFrame()
            intermediate_df["sizeOfTrainingSet"]       = list(range(len(roundMae)))
            intermediate_df["averageError"]            = roundMae
            intermediate_df["stdErrorOfMeanError"]     = roundMae
            intermediate_df["stdDeviationOfMeanError"] = roundMae
            intermediate_df.to_csv(os.path.join(roundPath,"full_round_" + str(inner_round) + ".csv"), index=False)

        total_elapsed_time = time.time() - optimization_start_time
        self.logger.info(self.logPrefix,
                    "Execution Time Is: " + str(timedelta(seconds=total_elapsed_time)))
        
        self.logger.info(self.logPrefix,
                    "The kFold process takes " +
                    str(total_kfold_elapsed_time * 100 / total_elapsed_time) +
                    "%% of the total optimization time.")

        result_df = pd.DataFrame()
        result_df["sizeOfTrainingSet"]       = np.array([iCnt for iCnt in sorted(maePerTrainSize.keys()) ])
        result_df["averageError"]            = [ np.array(maePerTrainSize[iCnt]).mean() for iCnt in maePerTrainSize.keys() ]
        result_df["stdErrorOfMeanError"]     = [ np.array(maePerTrainSize[iCnt]).std() / math.sqrt(iCnt) for iCnt in maePerTrainSize.keys() ]
        result_df["stdDeviationOfMeanError"] = [ np.array(maePerTrainSize[iCnt]).std()  for iCnt in maePerTrainSize.keys() ]

        return result_df

class SerialOptimization(OptimizationFactory):
    def __init__(self, logger : Logger):
        self.logger = logger
        self.logPrefix = "Serial Optimization"
    
    def optimizeModel(self, model : any, zifs : pd.DataFrame, X_featureNames : list, Y_featureNames : list , designspace_thres : int, bo_selection : str, save_path : str) -> pd.DataFrame:

        """ Serial Optimization As A Method For Optimizing MAE of LogD 
            model:              The model to be optimized.
            zifs :              The data used during optimization.
            X_featureNames:     The names of the training features.
            Y_featureNames:     The names of the target features.
        """
        optimization_start_time = time.time()
        # Make a list with all unique ZIF names.
        uniqueZIFs = zifs.type.unique()

        # Count the total number that the kfold process takes in seconds
        total_kfold_elapsed_time = 0.0

        # Initialize dictionary of errors per training data size
        fold_num = 10
        design_space = (len(uniqueZIFs))
        fold_design_space = len(uniqueZIFs) * ((fold_num - 1) / fold_num)

        select_data_points_num = 0
        if design_space < designspace_thres:
            fold_num = len(uniqueZIFs)
            select_data_points_num = len(uniqueZIFs) - 1

        elif fold_design_space < designspace_thres:
            select_data_points_num = int(fold_design_space)

        else:
            select_data_points_num = designspace_thres

        self.logger.info(self.logPrefix,
                         "----------   Data points to be selected are: " + str(select_data_points_num) +  " out of a design space of " + str(design_space) + " data points." + "     ----------")


        zif_kfold = KFold(n_splits=fold_num, shuffle=True, random_state=10)
        inner_round = 0 
        maePerTrainSize = {}
        for train_zif_indicies, left_out_zif_indicies in zif_kfold.split(uniqueZIFs):
            inner_round += 1
            roundPath = os.path.join(save_path, "Round_" + str(inner_round))
            os.mkdir(roundPath)
            roundMae = []

            self.logger.info(self.logPrefix,
                        "----------   Round " + str(inner_round) + "     ----------")

            trainZIFnames = np.delete(uniqueZIFs, left_out_zif_indicies)
            testZIFname   = uniqueZIFs[left_out_zif_indicies]

            trainZIFs = zifs[~zifs['type'].isin(testZIFname)]
            testZIFs  = zifs[zifs['type'].isin(testZIFname)]

            selectRandomSample = 0
            currentData   = pd.DataFrame()

            maeBestPerformanceList      = []
            maeStopCriterionMet         = False
            bestPerformingData          = {}

            for sizeOfTrainZIFs in range(select_data_points_num):

                # Sample each ZIF serialy.
                serialSelection = SerialSelectionStrategy(logger=self.logger)
                selectedZifName = serialSelection.select_next_instance(uniqueZIFs, sizeOfTrainZIFs)
                selectedZIF     = trainZIFs[(trainZIFs['type'] == selectedZifName)]

                # Remove the sellected ZIF from the list of available for training
                trainZIFs     = trainZIFs[(trainZIFs['type']) != selectedZifName]
                trainZIFnames = np.delete(trainZIFnames, np.where(trainZIFnames == selectedZifName))

                selectRandomSample += 1

                # Add the next ZIF to the currently used data.
                currentData = pd.concat([currentData, selectedZIF], axis=0, ignore_index=True)

                # Create feature matrices for all currently used data.
                x_trainAll = currentData[X_featureNames].to_numpy()
                y_trainAll = currentData[Y_featureNames].to_numpy()

                # Prediction on outer leave one out test data
                x_test  = testZIFs[X_featureNames].to_numpy()
                y_test  = testZIFs[Y_featureNames].to_numpy()

                model.fit(x_trainAll, y_trainAll.ravel())

                y_pred  = model.predict(x_test)

                mae = metrics.mean_absolute_error(y_test, y_pred)

                saveCurrentData = False
                stopCriterion   = None
                if not maeStopCriterionMet:
                    if len(maeBestPerformanceList) == 5:
                        stopCriterion = "low_performance_gain"
                        maeStopCriterionMet = True
                    else:

                        if mae <= 0.5:
                            stopCriterion = "error_threshold_reached"
                            maeStopCriterionMet = True
                            saveCurrentData = True
                            bestPerformingData     = {}

                        elif (not maeBestPerformanceList) or (mae > maeBestPerformanceList[0] - (maeBestPerformanceList[0] * 0.1)):
                            maeBestPerformanceList.append(mae)
                            saveCurrentData = True

                        else:
                            maeBestPerformanceList = []
                            bestPerformingData     = {}

                    if saveCurrentData:
                        datasetsInDict = len(bestPerformingData) + 1
                        bestPerformingData[datasetsInDict] = currentData
                        bestPerformingData[datasetsInDict]["mae"] = mae
                        bestPerformingData[datasetsInDict]["tested_against"] = '-'.join(testZIFname)

                    if maeStopCriterionMet:
                        # Save th ecurrent snapshot of the data to a csv file. Use a random name to avoid overwriting
                        for key, value in bestPerformingData.items():
                            bestPerformingData[key]["stop_criterion"] = stopCriterion
                            bestPerformingDataName = "Round_" + str(inner_round) + "_" + "Dataset_No_" + str(key) + "_best_performing_dataset_of_size_" + str(len(value.type.unique())) + "_" + ''.join(random.choice(string.ascii_letters) for _ in range(5)) + ".csv"
                            bestPerformingData[key].to_csv(os.path.join(roundPath,bestPerformingDataName), index=False)


                if (sizeOfTrainZIFs + 1) not in maePerTrainSize.keys():
                    maePerTrainSize[(sizeOfTrainZIFs + 1)] = []
                
                # Append mae to the corresponding dictionary list
                maePerTrainSize[(sizeOfTrainZIFs + 1)].append(mae)

                self.logger.info(self.logPrefix, 
                            "Number of ZIFs in Dataset: " + str((sizeOfTrainZIFs + 1)))
                self.logger.info(self.logPrefix,
                            "Mean Absolute Error: " + str(mae))

                roundMae.append(mae)

            self.logger.info(self.logPrefix,"Writting results of Round " + str(inner_round) + " to file.")

            intermediate_df = pd.DataFrame()
            intermediate_df["sizeOfTrainingSet"]       = list(range(len(roundMae)))
            intermediate_df["averageError"]            = roundMae
            intermediate_df["stdErrorOfMeanError"]     = roundMae
            intermediate_df["stdDeviationOfMeanError"] = roundMae
            intermediate_df.to_csv(os.path.join(roundPath,"full_round_" + str(inner_round) + ".csv"), index=False)

        total_elapsed_time = time.time() - optimization_start_time
        self.logger.info(self.logPrefix,
                    "Execution Time Is: " + str(timedelta(seconds=total_elapsed_time)))
        
        self.logger.info(self.logPrefix,
                    "The kFold process takes " +
                    str(total_kfold_elapsed_time * 100 / total_elapsed_time) +
                    "%% of the total optimization time.")

        result_df = pd.DataFrame()
        result_df["sizeOfTrainingSet"]       = np.array([iCnt for iCnt in sorted(maePerTrainSize.keys()) ])
        result_df["averageError"]            = [ np.array(maePerTrainSize[iCnt]).mean() for iCnt in maePerTrainSize.keys() ]
        result_df["stdErrorOfMeanError"]     = [ np.array(maePerTrainSize[iCnt]).std() / math.sqrt(iCnt) for iCnt in maePerTrainSize.keys() ]
        result_df["stdDeviationOfMeanError"] = [ np.array(maePerTrainSize[iCnt]).std()  for iCnt in maePerTrainSize.keys() ]

        return result_df
