# Surrogate model imports
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import ConstantKernel

from xgboost import XGBRegressor

# Acquisition Function imports
from scipy.stats import norm

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

# File imports
import os


# Expected Improvement
def expected_improvement(x, trainLabels, model, best_y, factor = 2.0):

    """ Function Computign The Expected Improvement Of All Data Points in a Dataset"""

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


def plot_graph(frame1, frame2 = None, frame3 = None, label1 = '', label2 = '', label3 = '', on_off = 'False', x_min=0, x_max=75, y_min=0, y_max=10, 
               size='16', line=2.5, edge=2, axes_width = 2, tickWidth = 2, tickLength=12, 
            xLabel = '', yLabel ='', fileName = 'picture.png', marker_colors = ['g', 'r']):
    
    """ Plot the LogD Mean (MAE) - Size of Training Scores For up to Three Methods """
    
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

def plotDataExists():
    
    """ If Data Already Exists Plot Directly From Them """

    if not os.path.exists('random.csv'):
        return False

    if not os.path.exists('serial.csv'):
        return False

    if not os.path.exists('bo.csv'):
        return False
    
    random_results = pd.read_csv('random.csv')
    serial_results = pd.read_csv('serial.csv')
    bo_results     = pd.read_csv('bo.csv')

    plot_graph(bo_results,  random_results, serial_results, 'Bayesian Optimization', 'Random Order', 'Researcher Order' 'True',
             -1, 75, 0.5, 6.5, 18, 1.5, 2, 2, 2, 8,
             'Number of ZIFs in the training dataset', 'Mean absolute error of log$\it{D}$',
             'validation_DataSetSize.png', marker_colors=['y','g','r'])

    return True

def bayesianOptimization():

    """ Bayesian Optimization As A Method For Optimizing MAE of LogD """

    data_from_file = readData()
    
    Y = ["logD"]
    X = ['diameter','mass','ascentricF', 'kdiameter','ionicRad',
         'MetalNum','MetalMass','σ_1', 'e_1',
         'linker_length1', 'linker_length2', 'linker_length3',
         'linker_mass1', 'linker_mass2', 'linker_mass3',
         'func1_length', 'func2_length', 'func3_length', 
         'func1_mass', 'func2_mass', 'func3_mass']
    
    sortedData  = data_from_file.sort_values(X)

    # Make a list with all unique ZIF names.
    uniqueZIFs = sortedData.type.unique()

    XGBR = XGBRegressor(n_estimators=500, max_depth=5, eta=0.07, subsample=0.75, colsample_bytree=0.7, reg_lambda=0.4, reg_alpha=0.13,
                        n_jobs=6,
                        # nthread=6,
                        random_state=6410
                        )
    
    gp_model = GaussianProcessRegressor(kernel=ConstantKernel(1.0) * RBF(1.0))


    # Initialize dictionary of errors per training data size
    maePerTrainSize = {}
    for leaveOutZifIndex in range(len(uniqueZIFs)):
        
        print("----------   Round " + str(leaveOutZifIndex + 1) + "     ----------")

        trainZIFnames = np.delete(uniqueZIFs, leaveOutZifIndex)
        testZIFname   = uniqueZIFs[leaveOutZifIndex]

        trainZIFs = sortedData[sortedData['type'] != testZIFname]
        testZIFs  = sortedData[sortedData['type'] == testZIFname]

        selectRandomSample = True
        currentData   = pd.DataFrame()
        currentRunMae = []
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
                ei,eiName = expected_improvement(trainZIFs, X, gp_model, minMae,0.0)
                # ei,eiName = upperConfidenceBound(trainZIFs, X, XGBR)
                selectedZIF = trainZIFs[(trainZIFs['type'] == eiName)]

                # Remove the sellected ZIF from the list of available for training
                trainZIFs = trainZIFs[(trainZIFs['type']) != eiName]
                trainZIFnames = np.delete(trainZIFnames, np.where(trainZIFnames == eiName))

            # Add the next ZIF to the currently used data.
            currentData = pd.concat([currentData, selectedZIF], axis=0, ignore_index=True)

            x_train = currentData[X].to_numpy()
            y_train = currentData[Y].to_numpy()


            x_test  = testZIFs[X].to_numpy()
            y_test  = testZIFs[Y].to_numpy()

            XGBR.fit(x_train, y_train.ravel())

            y_pred  = XGBR.predict(x_test)

            # best_y = np.max(y_pred)
            # best_y = np.min(y_pred)

            mae = metrics.mean_absolute_error(y_test, y_pred)
            
            for i in range(selectedZIF.shape[0]):
                currentRunMae.append(mae)
            
            minMae = min(currentRunMae)

            # Fit the Gaussian process model to the sampled points
            gp_model.fit(x_train, np.array(currentRunMae))

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
   
    bo_result = bayesianOptimization()

    # bo_result      = pd.read_csv('bo.csv')
    random_results = pd.read_csv('random.csv')
    serial_results = pd.read_csv('serial.csv')

    plot_graph(bo_result, random_results, serial_results, 'Bayesian Optimization', 'Random Order','Researcher Order', 'True',
             -1, 75, 0.5, 6.5, 18, 1.5, 2, 2, 2, 8,
             'Number of ZIFs in the training dataset', 'Mean absolute error of log$\it{D}$',
             'validation_DataSetSize.png', marker_colors=['y', 'g', 'r'])