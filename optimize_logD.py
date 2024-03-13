import os
from datetime import datetime
import logging
import pandas as pd
from statistical_tests import Statistical_Tests
from ga_inverse import readData
from xgboost import XGBRegressor
from optimization_methods import BayesianOptimization
from plot_optimization import plot_logD_trainSize_perMethod

def plot_data_exists(data_path) -> bool:

    """ Check wheather plot data already exist and return the respective truth value.
        data_path1:     The path to look for the set of data."""

    if not os.path.exists(data_path):
        return False

    return True

def data_preparation(sourceFile=None) -> list:

    if sourceFile is not None:
        data_from_file = readData(sourceFile)
    else:
        data_from_file = readData()

    Y = ["logD"]
    X = ['diameter','mass','ascentricF', 'kdiameter','ionicRad',
         'MetalNum','MetalMass','σ_1', 'e_1',
         'linker_length1', 'linker_length2', 'linker_length3',
         'linker_mass1', 'linker_mass2', 'linker_mass3',
         'func1_length', 'func2_length', 'func3_length', 
         'func1_mass', 'func2_mass', 'func3_mass']
    
    return data_from_file, X, Y

if __name__ == "__main__":

    # currDateTime = datetime.now().strftime('Optimization_%d-%m-%Y-%H:%M')
    currDateTime = datetime.now().strftime('Optimization_%d-%m-%Y-%H-%M') # modified for Windows env. from PanKrok

    # Create a directory to store the results of the experiments
    resultsPath = os.path.join(os.curdir,"Experiments")
    if not os.path.exists(resultsPath):
        os.mkdir("Experiments")

    bayesianData = 'bo.csv'
    randomData   = 'random.csv'
    serialData   = 'serial.csv'

    # Create a specific results direcotry for this run of BO.
    curRunResultsPath = os.path.join(resultsPath,currDateTime)
    os.mkdir(curRunResultsPath)
    if plot_data_exists(bayesianData):
        bo_result = pd.read_csv(bayesianData)
    else:

        # Logging Configuration TODO Create a separate logger class.
        logger = logging.getLogger('BO_logger')
        logger.setLevel(level=logging.DEBUG)

        fileHandler  = logging.FileHandler(os.path.join(curRunResultsPath, currDateTime + ".log"))
        fileHandler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        logger.addHandler(fileHandler)

        zifs, featureNames, targetNames = data_preparation("./DataPerGas/H2.xlsx")

        # Instantiate the XGB regressor model
        XGBR = XGBRegressor(n_estimators=500, max_depth=5, eta=0.07, subsample=0.75, colsample_bytree=0.7, reg_lambda=0.4, reg_alpha=0.13,
                            n_jobs=6,
                            # nthread=6,
                            random_state=6410
                            )
        # Instantiate Bayesian Optimizer
        bayesianOpt = BayesianOptimization()

        # Get the optimized model
        bo_result = bayesianOpt.optimizeModel(XGBR, zifs, featureNames, targetNames,logger)

        bo_result.to_csv(os.path.join(curRunResultsPath,"bo.csv"), index=False)
    
    pairedtTest = Statistical_Tests("pairedT")

    random_results = None
    if plot_data_exists(randomData):
        random_results = pd.read_csv(randomData)
        test_result = pairedtTest.getTest(bo_result["averageError"].to_numpy(),random_results["averageError"].to_numpy())
        print("P-Value of Paired T Test Between Bayesian Optimzation and Random Order: " + str(test_result.pvalue))

    serial_results = None
    if plot_data_exists(serialData):
        serial_results = pd.read_csv(serialData)
        test_result = pairedtTest.getTest(bo_result["averageError"].to_numpy(),serial_results["averageError"].to_numpy())
        print("P-Value of Paired T Test Between Bayesian Optimzation and Serial Order: " + str(test_result.pvalue))

    plot_logD_trainSize_perMethod(bo_result, random_results, serial_results, 'Bayesian Optimization', 'Random Order','Researcher Order', 'True',
             -1, 75, 0.5, 6.5, 18, 1.5, 2, 2, 2, 8,
             'Number of ZIFs in the training dataset', 'Mean absolute error of log$\it{D}$',
             os.path.join(curRunResultsPath, "plot_LogD-#Training_Points.png"), marker_colors=['y', 'g', 'r'])