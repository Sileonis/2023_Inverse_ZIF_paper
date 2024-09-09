import os
import sys
import time
import argparse
import logging
from logger import Logger
import pandas as pd
from datetime import datetime
from statistical_tests import Statistical_Tests
from xgboost import XGBRegressor
from optimization_methods import BayesianOptimization
from optimization_methods import RandomOptimization
from optimization_methods import SerialOptimization
from plot_optimization import plot_logD_trainSize_perMethod

import os
import sys
sys.path.insert(0,os.curdir)
sys.path.insert(0,os.pardir)
from ga_inverse import readData

def plot_data_exists(data_path) -> bool:

    """ Check wheather plot data already exist and return the respective truth value.
        data_path1:     The path to look for the set of data."""

    if not os.path.exists(data_path):
        return False

    return True

def data_preparation(sourceFile=None, research_data="zifs_diffusivity") -> list:

    Y = []
    X = []

    if research_data == "zifs_diffusivity":
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
    
    elif research_data == "co2":
        data_from_file = pd.read_csv(sourceFile)
        data_from_file = data_from_file.rename(columns={'CO2_working_capacity(mol/kg)':'working_capacity', 'mof_name':'type'})

        # One Hot Encode Data
        features = ["Nodular_BB1", "Nodular_BB2", "Connecting_BB1", "Connecting_BB2"]
        data_from_file = pd.get_dummies(data_from_file, columns=features,dtype=int)

        Y = ["working_capacity"]
        X = [feature_label for base_label in features for feature_label in list(data_from_file.columns) if base_label in feature_label]

    elif research_data == "o2_n2":
        data_from_file = pd.read_csv(sourceFile)

        Y = ["logSelfD"]
        X = ['LCD',	'PLD',	'LFPD',	'Volume',	'ASA_m2_g',	
             'ASA_m2_cm3',	'NASA_m2_g', 'NASA_m2_cm3',	
             'AV_VF',	'AV_cm3_g',	'NAV_cm3_g', ' H', 'C',	'N', 'metal type', 
             ' total degree of unsaturation', 'metalic percentage',	' oxygetn-to-metal ratio',	
             'electronegtive-to-total ratio', ' weighted electronegativity per atom', 
             ' nitrogen to oxygen ', 'mass',	'ascentricF',	'diameter',	'kdiameter']

    elif research_data == "o2":
        data_from_file = pd.read_csv(sourceFile)

        Y = ["logD_O2"]
        X = ['LCD',	'PLD',	'LFPD',	'Volume',	'ASA_m2_g',	
             'ASA_m2_cm3',	'NASA_m2_g', 'NASA_m2_cm3',	
             'AV_VF',	'AV_cm3_g',	'NAV_cm3_g', ' H', 'C',	'N', 'metal type', 
             ' total degree of unsaturation', 'metalic percentage',	' oxygetn-to-metal ratio',	
             'electronegtive-to-total ratio', ' weighted electronegativity per atom', 
             ' nitrogen to oxygen ']

    elif research_data == "n2":
        data_from_file = pd.read_csv(sourceFile)

        Y = ["logD_N2"]
        X = ['LCD',	'PLD',	'LFPD',	'Volume',	'ASA_m2_g',	
             'ASA_m2_cm3',	'NASA_m2_g', 'NASA_m2_cm3',	
             'AV_VF',	'AV_cm3_g',	'NAV_cm3_g', ' H', 'C',	'N', 'metal type', 
             ' total degree of unsaturation', 'metalic percentage',	' oxygetn-to-metal ratio',	
             'electronegtive-to-total ratio', ' weighted electronegativity per atom', 
             ' nitrogen to oxygen ']


    else:
        data_from_file = pd.read_csv(sourceFile)
        data_from_file = data_from_file.rename(columns={' absolute methane uptake high P [v STP/v]':'methane_uptake', ' name':'type'})        

        Y = ['methane_uptake']
        X = ['dimensions', ' supercell volume [A^3]', ' density [kg/m^3]',
             ' surface area [m^2/g]', ' num carbon', ' num hydrogen',
             ' num nitrogen', ' num oxygen', ' num sulfur', ' num silicon',
             ' vertices', ' edges', ' genus', ' largest included sphere diameter [A]',
             ' largest free sphere diameter [A]', ' largest included sphere along free sphere path diameter [A]']

    return data_from_file, X, Y

if __name__ == "__main__":

    # Command line parameters
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--data',     help='A file containing the train data.', default='TrainData.xlsx')
    parser.add_argument('-t', '--type',     help='The research data type. One of [zifs_diffusivity, co2, o2_n2, methane].', default='zifs_diffusivity')
    parser.add_argument('-m', '--method',   help='Select the optimization method to be used one of [bo, random, serial].', default='bo')
    parser.add_argument('-n', '--number',   help='The number of data points that will be selected from the design space', default=100)
    parser.add_argument('-b', '--bayesian', help='A file containing the logD data acquired by adding zifs using the bayesian optimization mehtod.', default='bo.csv')
    parser.add_argument('-r', '--random',   help='A file containing the logD data acquired by adding zifs in random order.', default='random.csv')
    parser.add_argument('-s', '--serial',   help='A file containing the logD data acquired by adding zifs in a specific serial order.', default='serial.csv')
    parser.add_argument('-o', '--output',   help='Whether the outpout should be printed on a stdout or a file or both.', default='filestream')
    parser.add_argument('-f', '--folder',   help='A folder with a signature name concerning the experiment conducted', default='test_opt')
    parser.add_argument('-l', '--loop',     help='Define the number of times the experiment should be conducted.', default=1)
    parser.add_argument('--bo_selection',   help='Define the selection method to be used durin the bayesian optimization. One of [greedy, prob].', default='prob')
    parsed_args = parser.parse_args() # Actually parse

    trainData         = parsed_args.data
    dataType          = parsed_args.type
    bayesianData      = parsed_args.bayesian
    randomData        = parsed_args.random
    serialData        = parsed_args.serial
    output            = parsed_args.output
    method            = parsed_args.method
    experiment_dir    = parsed_args.folder
    bo_selection      = parsed_args.bo_selection
    designspace_thres = int(parsed_args.number)
    experiments_num   = int(parsed_args.loop)

    if dataType not in ["zifs_diffusivity", "co2", "o2_n2", "methane"]:
        raise Exception("Invalid research data type.")

    # Create a directory to store the results of the experiments
    resultsPath = os.path.join("../","Experiments")
    if not os.path.exists(resultsPath):
        os.mkdir(resultsPath)

    # Create a specific results directory for this experiment
    for directory in experiment_dir.split(os.path.sep):
    
        resultsPath = os.path.join(resultsPath, directory)

        if not os.path.exists(resultsPath):
            os.mkdir(resultsPath)


    for i in range(experiments_num):

        log_filename = datetime.now().strftime('Optimization_%d-%m-%Y-%H-%M-%S.%f')[:-3]
        
        if method == "bo":
            log_filename = "Bayesian_" + log_filename
        elif method == "random":
            log_filename = "Random_" + log_filename
        elif method == "serial":
            log_filename = "Serial_" + log_filename
        else:
            raise Exception("Invalid optimization method.")
        

        # Create a specific results directory for this run of BO.
        curRunResultsPath = os.path.join(resultsPath, log_filename)
        os.mkdir(curRunResultsPath)

        # Create a specific directory for the intermediate saved datasets
        savedDataPath = os.path.join(curRunResultsPath, "saved_datasets")
        os.mkdir(savedDataPath)

        logger = Logger(name = 'BO_logger', level=logging.DEBUG, output=output,
                        filePath=os.path.join(curRunResultsPath, log_filename + ".log"))


        logger.info("Optimization", "Experiment " + str(i + 1) + " of " + str(experiments_num) + " started.")
        
        if plot_data_exists(bayesianData):
            result = pd.read_csv(bayesianData)
        else:

            np_data, featureNames, targetNames = data_preparation(trainData,dataType)

            # Instantiate the XGB regressor model
            XGBR = XGBRegressor(n_estimators=500, max_depth=5, eta=0.07, subsample=0.75, colsample_bytree=0.7, reg_lambda=0.4, reg_alpha=0.13,
                                random_state=6410
                                )
            # Instantiate An Optimizer
            optimizer   = None
            result_name = None
            if method == 'bo':
                optimizer = BayesianOptimization(logger)
                result_name = 'bo.csv'
            elif method == 'random':
                optimizer = RandomOptimization(logger)
                result_name = 'random_opt.csv'
            elif method == 'serial':
                optimizer = SerialOptimization(logger)
                result_name = 'serial_opt.csv'
            else:
                raise NotImplementedError("Invalid optimization method provided.")

            # Get the optimized model
            result = optimizer.optimizeModel(XGBR, np_data, featureNames, targetNames, designspace_thres, bo_selection ,savedDataPath)

            result.to_csv(os.path.join(curRunResultsPath,result_name), index=False)
        
        pairedtTest = Statistical_Tests("pairedT", logger)

        if (not plot_data_exists(randomData)) and (not plot_data_exists(serialData)):
            plot_logD_trainSize_perMethod(frame1=result, label1='Bayesian Optimization', on_off='True',
                                        xLabel='Number of ZIFs in the training dataset', yLabel='Mean absolute error of logD',
                                        fileName=os.path.join(curRunResultsPath, "plot_LogD-#Training_Points.png"), marker_colors=['y'])

        random_results = None
        bo_v_random_stats = None
        if plot_data_exists(randomData):
            random_results = pd.read_csv(randomData)
            stat_test = pairedtTest.getTest(result["averageError"].to_numpy(),random_results["averageError"].to_numpy())
            bo_v_random_stats = {"pvalue": stat_test.pvalue, "statistic": stat_test.statistic}
            print("P-Value of Paired T Test Between Bayesian Optimzation and Random Order: " + str(stat_test.pvalue))
            print("Statistic Value: " + str(stat_test.statistic))

            plot_logD_trainSize_perMethod(frame1=result, frame2=random_results, method1_v_method2_stats=bo_v_random_stats, label1='Bayesian Optimization', label2='Random Order', on_off='True',
                                        xLabel='Number of ZIFs in the training dataset', yLabel='Mean absolute error of logD',
                                        fileName=os.path.join(curRunResultsPath, "plot_LogD-#Training_Points.png"), marker_colors=['y', 'g'])

        serial_results = None
        bo_v_serial_stats = None
        if plot_data_exists(serialData):
            serial_results = pd.read_csv(serialData)
            stat_test = pairedtTest.getTest(result["averageError"].to_numpy(),serial_results["averageError"].to_numpy())
            bo_v_serial_stats = {"pvalue": stat_test.pvalue, "statistic": stat_test.statistic}
            print("P-Value of Paired T Test Between Bayesian Optimzation and Serial Order: " + str(stat_test.pvalue))

            plot_logD_trainSize_perMethod(frame1=result, frame2=serial_results, method1_v_method2_stats=bo_v_serial_stats, label1='Bayesian Optimization', label2='Serial Order', on_off='True',
                                        xLabel='Number of ZIFs in the training dataset', yLabel='Mean absolute error of logD',
                                        fileName=os.path.join(curRunResultsPath, "plot_LogD-#Training_Points.png"), marker_colors=['y', 'r'])
