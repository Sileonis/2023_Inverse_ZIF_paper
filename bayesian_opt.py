# Surrogate model imports
from sklearn.gaussian_process import GaussianProcessRegressor

from xgboost import XGBRegressor

# Acquisition Function imports
from scipy.stats import norm


# Metric imports
from sklearn import metrics
import math


# Data handling imports
import numpy as np
import pandas as pd



from ga_inverse import readData

# Plot imports

import matplotlib.pyplot as plt


# Expected Improvement
def expected_improvement(x, trainLabels, model, best_y):
    x_train = x[trainLabels].to_numpy()

    y_pred = model.predict(x_train)

    y_std  = y_pred.std()

    z = np.divide(np.subtract(y_pred, best_y), y_std)
    ei = (np.subtract(y_pred, best_y) * norm.cdf(z)) + (y_std * norm.pdf(z))

    return ei, x.iloc[np.argmax(ei)]['type']

def main():
    
    # Read the data
    data_from_file = readData()
    # print(data_from_file)    
    
    # Train the surrogate model (Select Mean and Covariance Functions)
    Y = ["logD"]
    # X = [i for i in data_labels if i not in Y]
    X = ['diameter','mass','ascentricF', 'kdiameter','ionicRad',
         'MetalNum','MetalMass','σ_1', 'e_1',
         'linker_length1', 'linker_length2', 'linker_length3',
         'linker_mass1', 'linker_mass2', 'linker_mass3',
         'func1_length', 'func2_length', 'func3_length', 
         'func1_mass', 'func2_mass', 'func3_mass']
    
    sortedData  = data_from_file.sort_values(X)

    uniqueZIFs = sortedData.type.unique()

    # zif67Data = sortedData[sortedData['type'] == uniqueZIFs[np.where(uniqueZIFs=='ZIF-67')[0][0]]]
    # beif1Data = sortedData[sortedData['type'] == uniqueZIFs[np.where(uniqueZIFs=='BeIF-1')[0][0]]]

    sortedDataX = sortedData[X]
    sortedDataY = sortedData[Y]

    XGBR = XGBRegressor(n_estimators=500, max_depth=5, eta=0.07, subsample=0.75, colsample_bytree=0.7, reg_lambda=0.4, reg_alpha=0.13,
                        n_jobs=6,
                        # nthread=6,
                        random_state=6410
                        )

    # Initialize dictionary of errors per training data size
    maePerTrainSize = {}

    selectRandomSample = True
    for leaveOutZifIndex in range(len(uniqueZIFs)):
        
        print("----------   Round " + str(leaveOutZifIndex + 1) + "     ----------")

        trainZIFnames = np.delete(uniqueZIFs, leaveOutZifIndex)
        testZIFname   = uniqueZIFs[leaveOutZifIndex]

        trainZIFs = sortedData[sortedData['type'] != testZIFname]
        testZIFs  = sortedData[sortedData['type'] == testZIFname]

        currentData = pd.DataFrame()
        sizeOfTrainData = 0
        for trainZIFIndex in range(len(uniqueZIFs) - 1):

            sizeOfTrainData += 1

            if selectRandomSample:
                # Sample 1 random ZIFs.
                # TODO Experiment with different multitudes of randome sampling
                print("Sampling 1 random ZIF.")

                startingZIF  = np.random.choice(trainZIFnames, size=1, replace=False)[0]
                selectedZIF  = trainZIFs[(trainZIFs['type'] == startingZIF)]

                # Remove the sellected ZIF from the list of available for training
                trainZIFs     = trainZIFs[(trainZIFs['type']) != startingZIF]
                trainZIFnames = np.delete(trainZIFnames, np.where(trainZIFnames == startingZIF))

                selectRandomSample = False
            else:
                # TODO Check the validity of the expected improvement function
                ei,eiName = expected_improvement(trainZIFs, X, XGBR, best_y)
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

            best_y = np.max(y_pred)

            mae = metrics.mean_absolute_error(y_test, y_pred)

            if sizeOfTrainData not in maePerTrainSize.keys():
                maePerTrainSize[sizeOfTrainData] = []
            
            # Append mae to the corresponding dictionary list
            maePerTrainSize[sizeOfTrainData].append(mae)

            print("Number of ZIFs in Dataset: " + str(sizeOfTrainData))
            print("Mean Average Error: " + str(mae))


    result_df = pd.DataFrame()
    result_df["sizeOfTrainingSet"] = np.array([iCnt for iCnt in sorted(maePerTrainSize.keys()) ])
    result_df["averageError"] = [ np.array(maePerTrainSize[iCnt]).mean() for iCnt in maePerTrainSize.keys() ]
    result_df["stdErrorOfMeanError"] = [ np.array(maePerTrainSize[iCnt]).std() / math.sqrt(iCnt) for iCnt in maePerTrainSize.keys() ]



            # sortedSampledX    = sortedSampledData[X]
            # sortedSampledY    = sortedSampledData[Y]

            # Train Surrogate Model
            # gpr = GaussianProcessRegressor(random_state=0).fit(sortedSampledX.to_numpy(), sortedSampledY.to_numpy())


            # # y_pred, y_std = gpr.predict(sortedDataX.to_numpy(), return_std=True)
            # y_pred = XGBR.predict(sortedDataX.to_numpy())


            # # Determine the point with the highest observed function value
            # best_y = starting_data[Y].max()

            # # Get an Acquisition Function in order to select the next sample
            # # ei,maxIndex = expected_improvement(sortedDataX.to_numpy(), gpr, best_y.to_numpy())

            # next_point = sortedDataX.to_numpy()[maxIndex]
            # next_pointA = sortedData.iloc[maxIndex]

            # sortedSampledData = pd.concat([sortedSampledData,next_pointA.to_frame().T])

            # mae = metrics.mean_absolute_error(sortedDataY.to_numpy(), y_pred)
            # print("Iteration " + str(i) + ": Mean Absolute Error: " + str(mae))

    plot_graph(result_df, 'Bayesian Optimization', '-', 'True',
             -1, 75, 0.5, 6.5, 18, 1.5, 2, 2, 2, 8,
             'Number of ZIFs in the training dataset', 'Mean absolute error of log$\it{D}$',
             'validation_DataSetSize.png', marker_colors=['r', 'g'])

def plot_graph(frame1, label1 = '', label2 = '', on_off = 'False', x_min=0, x_max=75, y_min=0, y_max=10, 
               size='16', line=2.5, edge=2, axes_width = 2, tickWidth = 2, tickLength=12, 
            xLabel = '', yLabel ='', fileName = 'picture.png', marker_colors = ['g', 'r']):
    # x = range(len(result_df))
    x1 = frame1['sizeOfTrainingSet']
    y1 = frame1['averageError']
    error1 = frame1['stdErrorOfMeanError']

    # plt.scatter(x, y)
    plt.errorbar(x1, y1, yerr=error1, label=label1, ecolor='k', fmt='o', c=marker_colors[0], markersize=size, linewidth=line, markeredgecolor='k', markeredgewidth=edge)
    # plt.yscale("log")
    plt.xlabel(xLabel, fontsize=size)
    plt.ylabel(yLabel, fontsize=size)
    # plt.title('Mean absolute error with error bars', fontsize=18)
    plt.rcParams["figure.figsize"] = (8,6)
    plt.legend(loc='upper left', fontsize=15, frameon=on_off)

    plt.tick_params(which='both', width=tickWidth)
    plt.tick_params(which='major', length=tickLength)
    plt.yticks(fontsize=size)
    plt.xticks(fontsize=size)
    plt.rcParams['axes.linewidth'] = axes_width

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    plt.savefig(fileName, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()
