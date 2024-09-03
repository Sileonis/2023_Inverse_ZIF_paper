import utils
import dialogs
import argparse
from functions import *

import os
import sys
sys.path.insert(0,os.curdir)
sys.path.insert(0,os.pardir)

from ga_inverse import readData

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path',      help='The path to the results of the experiments.', default='./')
    parser.add_argument('-t', '--train',     help='The path to the original training data.', default='')
    parser.add_argument('-s', '--save',      help='The name of the directory containing round results.', default='saved_datasets/')

    parsed_args = parser.parse_args() # Actually parse

    path       = parsed_args.path
    saveName   = parsed_args.save
    train_path = parsed_args.train
    sortedDataSizeFreq, stopDataSizeFreqThres, stopDataSizeFreqPerf, mostFreqDataSize, thresholdReachingZifs, lowPerformanceZifs, full_result_thresh, total_runs, max_dataset_size = parse_data(path,saveName)

    if train_path == '':        
        if os.path.exists("./train_data"):
            train_path = "./train_data/train_zifs_diffusivity/TrainData.xlsx"
        elif os.path.exists("../train_data"):
            train_path = "../train_data/train_zifs_diffusivity/TrainData.xlsx"
        else:
            raise Exception("Training data not found in default paths. Please provide a path to the original training data using the -t argument.")

    train_data = readData(train_path)

    action = 0
    while action != dialogs.getNumOfActions():

        utils.printEmptyLine()
        action = dialogs.selectMainAction()

        match action:

            case 1: # Plot all results
                utils.printEmptyLine()

                plot_mae_per_size(sortedDataSizeFreq)

            case 2: # Plot results for threshold criterion
                utils.printEmptyLine()

                plot_mae_per_size(stopDataSizeFreqThres)

            case 3: # Plot results for performance criterion
                utils.printEmptyLine()

                plot_mae_per_size(stopDataSizeFreqPerf)

            case 4: # Analyse statistics by data size
                utils.printEmptyLine()

                analyse_by_data_size(mostFreqDataSize, path, saveName)

            case 5: # Plot cumulative probability of achieving theshold
                utils.printEmptyLine()

                cumulative_thres(full_result_thresh,total_runs,max_dataset_size)
                # cumulative_thres(stopDataSizeFreqThres,total_runs)

            case 6 : # Plot success rate per zif
                utils.printEmptyLine()
                
                succes_rate_per_zif(thresholdReachingZifs, lowPerformanceZifs)
            
            case 7 : # Print all datasets used against a zif
                utils.printEmptyLine()

                datasets_used_against_zif(thresholdReachingZifs, lowPerformanceZifs)
            
            case 8: # Gather datasets by probability
                utils.printEmptyLine()
                
                get_datasets_by_probability(train_data, full_result_thresh, total_runs,max_dataset_size)

            case 9: # Gather datasets by probability
                utils.printEmptyLine()
                
                get_datasets_by_data_size(train_data, full_result_thresh, total_runs,max_dataset_size)

            case 10: # Gather datasets by probability
                utils.printEmptyLine()
                
                test_against_all_zifs(train_data, full_result_thresh, total_runs)

            case 11: # Plot error to dataset size for all runs.
                utils.printEmptyLine()
                
                plot_error_to_dataset_size(path)
            
            case 12:
                utils.printEmptyLine()

                test_top100_mofs_prediction(path)

            case _:
                pass