# Surrogate model imports
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import ConstantKernel
from xgboost import XGBRegressor

# Acquisition functions
from acquisition_functions import ExpectedImprovementCalculator
from selection_strategy    import GreedySelectionStrategy

# Metric imports
from sklearn import metrics
import math
import random

# Data handling imports
import numpy as np
import pandas as pd

# Code imports
from ga_inverse import readData

# Plot imports
import matplotlib.pyplot as plt

# Filesystem imports
import os


def plot_logD_trainSize_perMethod(frame1, frame2 = None, frame3 = None, label1 = '', label2 = '', label3 = '', on_off = 'False', x_min=0, x_max=75, y_min=0, y_max=10, 
               size='16', line=2.5, edge=2, axes_width = 2, tickWidth = 2, tickLength=12, 
            xLabel = '', yLabel ='', fileName = 'picture.png', marker_colors = ['y', 'g', 'r']):
    
    """ Plot the Mean (MAE) of logD (y-axis) to Size of Training Dataset (x-axis) for up to three methods 
        frame1 - 3:     A dataframe containing the follwing Columns:
                                                                    1.  sizeOfTrainingSet
                                                                    2.  averageError
                                                                    3.  stdErrorOfMeanError
        label1 - 3:     The name of the respective method used.
        on_off:         The value of frameon argument for pyplot.legend function.
        x_min:          The minimum value of x-axis
        x_max:          The maximum value of x-axis
        y_min:          The minimum value of y-axis
        y_max:          The maximum value of y-axis
        xLabel:         The label of x-axis
        yLabel:         The label of y-axis
        fileName:       The name under which the plot will be saved.
        marker_colors:  The colors that distinguish each method.
        """
    
    # First Method
    x1 = frame1['sizeOfTrainingSet']
    y1 = frame1['averageError']
    error1 = frame1['stdErrorOfMeanError']
    plt.errorbar(x1, y1, yerr=error1, label=label1, ecolor='k', fmt='o', c=marker_colors[0], markersize=size, linewidth=line, markeredgecolor='k', markeredgewidth=edge)

    # Second Method
    if frame2 is not None:
        x2 = frame2['sizeOfTrainingSet']
        y2 = frame2['averageError']
        error2 = frame2['stdErrorOfMeanError']
        plt.errorbar(x2, y2, yerr=error2, label=label2, ecolor='k', fmt='o', c=marker_colors[1], markersize=size, linewidth=line, markeredgecolor='k', markeredgewidth=edge)

    # Third Method
    if frame3 is not None:
        x3 = frame3['sizeOfTrainingSet']
        y3 = frame3['averageError']
        error3 = frame3['stdErrorOfMeanError']
        plt.errorbar(x3, y3, yerr=error3, label=label3, ecolor='k', fmt='o', c=marker_colors[2], markersize=size, linewidth=line, markeredgecolor='k', markeredgewidth=edge)


    plt.xlabel(xLabel, fontsize=size)
    plt.ylabel(yLabel, fontsize=size)
    # plt.title('Mean absolute error with error bars', fontsize=18)
    plt.rcParams["figure.figsize"] = (8,6)
    plt.legend(loc='upper right', fontsize=15, frameon=on_off)

    plt.tick_params(which='both', width=tickWidth)
    plt.tick_params(which='major', length=tickLength)
    plt.yticks(fontsize=size)
    plt.xticks(fontsize=size)
    plt.rcParams['axes.linewidth'] = axes_width

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    plt.savefig(fileName, bbox_inches='tight')
    plt.show()

def plot_data_exists(data_path) -> bool:

    """ Check wheather plot data already exist and return the respective truth value.
        data_path1:     The path to look for the set of data."""

    if not os.path.exists(data_path):
        return False

    return True

def data_preparation() -> list:

    data_from_file = readData()
    
    Y = ["logD"]
    X = ['diameter','mass','ascentricF', 'kdiameter','ionicRad',
         'MetalNum','MetalMass','σ_1', 'e_1',
         'linker_length1', 'linker_length2', 'linker_length3',
         'linker_mass1', 'linker_mass2', 'linker_mass3',
         'func1_length', 'func2_length', 'func3_length', 
         'func1_mass', 'func2_mass', 'func3_mass']
    
    return data_from_file, X, Y

def bayesianOptimization(zifs : pd.DataFrame, X_featureNames : list, Y_featureNames : list) -> pd.DataFrame:

    """ Bayesian Optimization As A Method For Optimizing MAE of LogD """

    # Make a list with all unique ZIF names.
    uniqueZIFs = zifs.type.unique()

    # Initiate an XGB regressor model
    XGBR = XGBRegressor(n_estimators=500, max_depth=5, eta=0.07, subsample=0.75, colsample_bytree=0.7, reg_lambda=0.4, reg_alpha=0.13,
                        n_jobs=6,
                        # nthread=6,
                        random_state=6410
                        )
    # Initiate a gaussian process model
    gp_model = GaussianProcessRegressor(kernel=ConstantKernel(1.0) * RBF(1.0))


    # Initialize dictionary of errors per training data size
    maePerTrainSize = {}
    for leaveOutZifIndex in range(len(uniqueZIFs)):
        
        print("----------   Round " + str(leaveOutZifIndex + 1) + "     ----------")

        trainZIFnames = np.delete(uniqueZIFs, leaveOutZifIndex)
        testZIFname   = uniqueZIFs[leaveOutZifIndex]

        trainZIFs = zifs[zifs['type'] != testZIFname]
        testZIFs  = zifs[zifs['type'] == testZIFname]

        selectRandomSample = True
        currentData   = pd.DataFrame()
        currentBayesianMae = []
        for sizeOfTrainZIFs in range(len(uniqueZIFs) - 1):

            if selectRandomSample:
                # Sample 1 random ZIFs.
                startingZIF  = np.random.choice(trainZIFnames, size=1, replace=False)[0]
                selectedZIF  = trainZIFs[(trainZIFs['type'] == startingZIF)]

                # Remove the sellected ZIF from the list of available for training
                trainZIFs     = trainZIFs[(trainZIFs['type']) != startingZIF]
                trainZIFnames = np.delete(trainZIFnames, np.where(trainZIFnames == startingZIF))

                selectRandomSample = False
            else:
                # Calculate the expected improvement values for all candidate zifs
                eiCalculator = ExpectedImprovementCalculator(factor=0)
                eI = eiCalculator.get_acquisition_function(trainZIFs, X_featureNames, gp_model, minMae,0.0)

                # Select the next zif in a greedy manner
                greedySelection = GreedySelectionStrategy()
                eiName = greedySelection.select_next_instance(eI, trainZIFs)
                selectedZIF = trainZIFs[(trainZIFs['type'] == eiName)]

                # Remove the sellected ZIF from the list of available for training
                trainZIFs = trainZIFs[(trainZIFs['type']) != eiName]
                trainZIFnames = np.delete(trainZIFnames, np.where(trainZIFnames == eiName))

            # Add the next ZIF to the currently used data.
            currentData = pd.concat([currentData, selectedZIF], axis=0, ignore_index=True)

            x_train = currentData[X_featureNames].to_numpy()
            y_train = currentData[Y_featureNames].to_numpy()


            x_test  = testZIFs[X_featureNames].to_numpy()
            y_test  = testZIFs[Y_featureNames].to_numpy()

            # Bayesian Leave One Out
            trainZifNames = trainZIFs.type.unique()
            bayesianTrainLength = len(trainZIFnames)
            if bayesianTrainLength > 1:
                bayesianAverageMAE = 0
                for excludedZifIndex in range(bayesianTrainLength):
                    bayesianTrainZIFnames = np.delete(trainZifNames, excludedZifIndex)
                    bayesianTestZIFname   = trainZifNames[excludedZifIndex]

                    bayesianTrainZIFs = zifs[zifs['type'] != bayesianTestZIFname]
                    bayesianTestZIF   = zifs[zifs['type'] == bayesianTestZIFname]

                    bayesianX_train   = bayesianTrainZIFs[X_featureNames].to_numpy()
                    bayesianY_train   = bayesianTrainZIFs[Y_featureNames].to_numpy()

                    bayesianX_test    = bayesianTestZIF[X_featureNames].to_numpy()
                    bayesianY_test    = bayesianTestZIF[Y_featureNames].to_numpy()

                    XGBR.fit(bayesianX_train, bayesianY_train.ravel())

                    bayesianY_pred    = XGBR.predict(bayesianX_test)

                    bayesianAverageMAE += metrics.mean_absolute_error(bayesianY_test,bayesianY_pred)

                bayesianAverageMAE /= bayesianTrainLength

            for i in range(selectedZIF.shape[0]):
                currentBayesianMae.append(bayesianAverageMAE)
            
            minMae = min(currentBayesianMae)

            # Prediction on outer leave one out test data
            XGBR.fit(x_train, y_train.ravel())

            y_pred  = XGBR.predict(x_test)

            # best_y = np.max(y_pred)
            # best_y = np.min(y_pred)

            mae = metrics.mean_absolute_error(y_test, y_pred)

            # Fit the Gaussian process model to the sampled points
            gp_model.fit(x_train, np.array(currentBayesianMae))

            if (sizeOfTrainZIFs + 1) not in maePerTrainSize.keys():
                maePerTrainSize[(sizeOfTrainZIFs + 1)] = []
            
            # Append mae to the corresponding dictionary list
            maePerTrainSize[(sizeOfTrainZIFs + 1)].append(mae)

            print("Number of ZIFs in Dataset: " + str((sizeOfTrainZIFs + 1)))
            print("Mean Average Error: " + str(mae))


    result_df = pd.DataFrame()
    result_df["sizeOfTrainingSet"] = np.array([iCnt for iCnt in sorted(maePerTrainSize.keys()) ])
    result_df["averageError"] = [ np.array(maePerTrainSize[iCnt]).mean() for iCnt in maePerTrainSize.keys() ]
    result_df["stdErrorOfMeanError"] = [ np.array(maePerTrainSize[iCnt]).std() / math.sqrt(iCnt) for iCnt in maePerTrainSize.keys() ]

    result_df.to_csv("bo.csv", index=False)

    return result_df

if __name__ == "__main__":

    bayesianData = 'bo.csv'
    randomData   = 'random.csv'
    serialData   = 'serial.csv'

    if plot_data_exists(bayesianData):
        bo_result = pd.read_csv(bayesianData)
    else:
        zifs, featureNames, targetNames = data_preparation()
        bo_result = bayesianOptimization(zifs, featureNames, targetNames)
    
    random_results = None
    if plot_data_exists(randomData):
        random_results = pd.read_csv(randomData)

    serial_results = None
    if plot_data_exists(serialData):
        serial_results = pd.read_csv(serialData)

    plot_logD_trainSize_perMethod(bo_result, random_results, serial_results, 'Bayesian Optimization', 'Random Order','Researcher Order', 'True',
             -1, 75, 0.5, 6.5, 18, 1.5, 2, 2, 2, 8,
             'Number of ZIFs in the training dataset', 'Mean absolute error of log$\it{D}$',
             'validation_DataSetSize.png', marker_colors=['y', 'g', 'r'])