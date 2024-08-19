import os
import dialogs
import shutil
import random
import string
import statistics
import numpy as np
from scipy import stats
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from colorama import init, Fore, Style


import os
import sys
sys.path.insert(0,os.path.join(os.pardir,"Bayesian_Optimization"))
from optimize_logD import data_preparation
from Bayesian_Optimization import plot_optimization

def parse_data(path: str, saveName: str = 'saved_datasets'):
    """
    Parses data from the specified directory and saves the results.

    Parameters:
        path     (str): The path to the directory containing the data.
        saveName (str): The name of the directory containing the saved data.

    Returns:
        sortedDataSizeFreq    (dict): A dictionary (key: data sizes value: frequency of stopping).
        stopDataSizeFreqThres (dict): A dictionary (key: data sizes value: frequency of reaching stopping due to reach of the error).
        stopDataSizeFreqPerf  (dict): A dictionary (key: data sizes value: frequency of reaching stopping due to low performance gain criterion).
        mostFreqDataSize      (int) : The most frequently encountered data size.
        thresholdReachingZifs (dict): A dictionary (key: tested ZIFs value: dictionaries containing the count of stopping due to error threshold reached and the datasets tested against each ZIF).
        lowPerformanceZifs    (dict): A dictionary (key: tested ZIFs value: dictionaries containing the count of stopping due to low performance and the datasets tested against each ZIF).
        full_result_thresh    (dict): A dictionary (key: data sizes value: frequency of reaching the error threshold for the full round).
        total_runs            (int) : The total number of runs performed.
    """
    
    total_runs = 0
    max_dataset_size = 0
    full_result_thresh = {}
    stopDataSizeFreq = {}
    stopDataSizeFreqThres = {}
    stopDataSizeFreqPerf = {}
    lowPerformanceZifs = {}
    thresholdReachingZifs = {}

    print("Please insert the error threshold that you wish to use for the full report analysis.")
    full_threshold = float(input())

    for experimentFile in os.listdir(path):

        # Skip non-directory files
        if not os.path.isdir(os.path.join(path, experimentFile)):
            continue

        savePath = os.path.join(path, experimentFile, saveName)
        if not os.path.isdir(savePath):
            continue

        for roundDir in os.listdir(savePath):

            # Skip non-directory files
            if not os.path.isdir(os.path.join(savePath, roundDir)):
                continue

            total_runs += 1

            selectedDataset = None
            for roundResult in os.listdir(os.path.join(savePath, roundDir)):

                fileSplit = roundResult.split('_')
                # Skip the full round report.
                if fileSplit[0] != 'full':

                    if (selectedDataset is None) or (fileSplit[-2] > selectedDataset.split('_')[-2]):
                        selectedDataset = os.path.join(savePath, roundDir, roundResult)
                else:
                    full_data_path = os.path.join(savePath, roundDir, roundResult)
                    full_data = pd.read_csv(full_data_path)

                    if not max_dataset_size:
                        for size, row in full_data.iterrows():
                            max_dataset_size += 1

                    for size, row in full_data.iterrows():
                        if row["averageError"] < full_threshold:
                            if (size + 1) not in full_result_thresh:
                                full_result_thresh[size + 1] = {"count": 1,
                                                                "datasets": [full_data_path]}
                            else:
                                full_result_thresh[size + 1]["count"] += 1
                                full_result_thresh[size + 1]["datasets"].append(full_data_path)
                            
                            break


            stoppedDataSize = selectedDataset.split('_')[-2]
            if stoppedDataSize in stopDataSizeFreq:
                stopDataSizeFreq[stoppedDataSize] += 1
            else:
                stopDataSizeFreq[stoppedDataSize] = 1

            data = pd.read_csv(selectedDataset)            
            tested_zif = data["tested_against"][0]
            if data["stop_criterion"][0] == "error_threshold_reached":
                if stoppedDataSize in stopDataSizeFreqThres:
                    stopDataSizeFreqThres[stoppedDataSize] += 1
                else:
                    stopDataSizeFreqThres[stoppedDataSize] = 1
                if tested_zif in thresholdReachingZifs:
                    thresholdReachingZifs[tested_zif]["count"] += 1
                    thresholdReachingZifs[tested_zif]["datasets"].append(list(data["type"].unique()))
                else:
                    thresholdReachingZifs[tested_zif] = {"count": 1,
                                                         "datasets": [list(data["type"].unique())]}
            else:
                if stoppedDataSize in stopDataSizeFreqPerf:
                    stopDataSizeFreqPerf[stoppedDataSize] += 1
                else:
                    stopDataSizeFreqPerf[stoppedDataSize] = 1
                
                if tested_zif in lowPerformanceZifs:
                    lowPerformanceZifs[tested_zif]["count"] += 1
                    lowPerformanceZifs[tested_zif]["datasets"].append(list(data["type"].unique()))
                else:
                    lowPerformanceZifs[tested_zif] = {"count": 1,
                                                      "datasets": [list(data["type"].unique())]}
                    

    sortedDataSizeFreqPerf  = {k: stopDataSizeFreqPerf[k] for k in sorted(stopDataSizeFreqPerf, key=lambda x: float(x))}
    sortedDataSizeFreq = {k: stopDataSizeFreq[k] for k in sorted(stopDataSizeFreq, key=lambda x: float(x))}
    mostFreqDataSize = list(sortedDataSizeFreq.keys())[0]

    number_of_stoped_runs = 0
    for key in thresholdReachingZifs.keys():
        number_of_stoped_runs += thresholdReachingZifs[key]["count"]
    for key in lowPerformanceZifs.keys():
        number_of_stoped_runs += lowPerformanceZifs[key]["count"]

    return sortedDataSizeFreq, stopDataSizeFreqThres, stopDataSizeFreqPerf, mostFreqDataSize, thresholdReachingZifs, lowPerformanceZifs, full_result_thresh, total_runs, max_dataset_size

def plot_mae_per_size(data: dict):
    """
    Plots a bar graph of the number of times the stop criteria were met for each number of Zifs in the dataset.

    Parameters:
        data (dict): A dictionary where the keys are the number of Zifs in the dataset and the values are the number of times the stop criteria were met.

    Returns:
        None
    """
    plt.bar(list(data.keys()), list(data.values()), width = 0.6)
    plt.xlabel('Number of Zifs in Dataset')
    plt.ylabel('Number of times stop criteria were met.')
    plt.show()

def box_plot_statistics(mae_data: list, case: str):
    """
    Generate a box plot and print describing statistics for the given data.

    Parameters:
        mae_data (list): A list of MAE values.
        case     (str) : A name specifying the plot.

    Returns:
        None
    """
    print("\n" + "Statistics for " + case + "\n")
    print("Minimum:", min(mae_data))
    print("Maximum:", max(mae_data))
    print("Median:",  statistics.median(mae_data))
    print("Mean:",    statistics.mean(mae_data))
    print("25th Percentile:", statistics.quantiles(mae_data, n=4)[0])
    print("75th Percentile:", statistics.quantiles(mae_data, n=4)[2])

    plt.boxplot(mae_data)
    plt.yticks(np.arange(0, max(mae_data), 0.5))
    plt.ylabel('Mean Absolute Error')
    plt.title('Box Plot of '+ case)
    plt.show()

def cumulative_thres(threshold_criterion_results: dict, numberOfRuns: int, max_dataset_size: int, plot: bool = True):
    """
    Calculate the cumulative possibility of activating the threshold criterion.

    Parameters:
        threshold_criterion_results (dict): A dictionary (key: data sizes, value: frequency of stopping due to threshold criterion).
        numberOfRuns                (int) : The total number of runs.
        plot                        (bool): Whether to plot the results. Default is True.

    Returns:
        dict or None: The sortedData dictionary if plot is False, otherwise None.
    """

    data_list = []
    data_name_list = []
    more_data = True
    temp_results = threshold_criterion_results.copy()
    while more_data:

        sortedData = {k: temp_results[k].copy() for k in sorted(temp_results, key=lambda x: float(x))}

        prevSum = 0
        for key in sortedData.keys():
            sortedData[key]["count"] += prevSum
            prevSum = sortedData[key]["count"]
            sortedData[key]["count"] /= numberOfRuns

        # In case a data size is missing use the vaulue from the previous one.
        for key in range(1, max_dataset_size + 1):
            if key not in sortedData.keys():
                if key > 1:
                    sortedData[key] = {"count": sortedData[key - 1]["count"]}
                else:
                    sortedData[key] = {"count": 0}
                sortedData[key]["datasets"] = []

        # Remove the first 5 data sizes which have been created randomly.
        for key in range(1,6):
            if key in sortedData.keys():
                del sortedData[key]

        print("What is the name for this dataset?")
        data_name_list.append(input())
        data_list.append(sortedData)

        print("Do you want to add more data to the cumulative probability plot? [Y/N]")
        if dialogs.yesNoInput():
            print("Insert the path for the new set of data.")
            new_data_path = input()

            parse_results = list(parse_data(new_data_path))

            temp_results.clear()
            temp_results     = parse_results[-3]
            numberOfRuns     = parse_results[-2]
            max_dataset_size = parse_results[-1]
        else:
            more_data = False

    if plot:
        for data in data_list:
            count_list = [data[size]["count"] for size in data.keys()]
            plt.scatter(data.keys(), count_list, label=data_name_list[data_list.index(data)])
        plt.grid()
        plt.legend()
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.show()

    return sortedData

def succes_rate_per_zif(success: dict, failure: dict):
    """
    Calculate the success rate for each ZIF.

    Parameters:
        success (dict): A dictionary (key: the tested ZIF, value: the count of successes (activation of threshold criterion) and the corresponding training datasets).
        failure (dict): A dictionary (key: the tested ZIF, value: the count of failures  (activation of low progression gain criterion) and the corresponding training datasets).

    Returns:
        Prints:
            The success rate for each ZIF, including the total count of successes and failures.
            The total number of ZIFs that achieved small error and the total number of ZIFs that achieved large error.
    """
        
    init()

    sorted_success = dict(sorted(success.items(), key=lambda x: x[1]["count"], reverse=True))

    tmp_failure = failure.copy()
    for key in sorted_success.keys():
        if key in tmp_failure.keys():
            print("ZIF: " + key + Fore.GREEN + " Success Rate: " + Style.RESET_ALL + str(sorted_success[key]["count"] / (sorted_success[key]["count"] + tmp_failure[key]["count"])) + "(" + str(sorted_success[key]["count"] + tmp_failure[key]["count"]) + ")")
            del tmp_failure[key]
        else:
            print("ZIF: " + key + Fore.GREEN + ", Succeded " + Style.RESET_ALL + str(sorted_success[key]["count"]) + " times and never failed.")

    for key in tmp_failure.keys():
        print("ZIF: " + key + Fore.RED + " Failed " + Style.RESET_ALL + str(tmp_failure[key]["count"]) + " times.")

    print("Total number of ZIFs that achieved small error: " + str(len(success)))
    print("Total number of ZIFs that did not achieve small  error: " + str(len(tmp_failure)))

def datasets_used_against_zif(success: dict, failure: dict):
    """
    Print the train datasets used against the selected ZIF.

    Parameters:
        success (dict): A dictionary (key: ZIF names, value: dictionaries containing the count of successes and the corresponding datasets).
        failure (dict): A dictionary (key: ZIF names, value: dictionaries containing the count of failures and the corresponding datasets).

    Returns:
        None

    """

    print("Enter zif name.")
    testZif = input()

    init()

    if testZif not in success.keys() and testZif not in failure.keys():
        print("The ZIF name you entered does not exist.")
    else:
        if testZif in success.keys():
            print(Fore.GREEN + "Succeded" + Style.RESET_ALL)
            for x in success[testZif]["datasets"]:
                print(x)

        if testZif in failure.keys():
            print(Fore.RED + "Failed" + Style.RESET_ALL)
            for x in failure[testZif]["datasets"]:
                print(x)    

def get_datasets(train_data: pd.DataFrame, cumulative_results: dict):
    """
    Gather the datasets by probability or data size.

    Parameters:
        train_data                  (pd.DataFrame) : The original training data used to reconstruct each intermediate dataset (zif by zif).
        cumulative_results          (dict)         : A dictionary (key: data sizes, value: frequency of stopping due to threshold criterion).
        selection_method            (str)          : The method used to select the datasets.
    Returns:
        None

    """   

    selected_datasets_path = os.path.join(os.curdir, "selected_datasets")
    if os.path.exists(selected_datasets_path):
        shutil.rmtree(os.path.join(selected_datasets_path,"",""))

    os.mkdir(selected_datasets_path)

    for key in cumulative_results.keys():
        for data_path in cumulative_results[key]["datasets"]:
            data_splits   = data_path.split(os.sep)
            data_round    = data_splits[-2].replace('_', ' ')
            log_file      = data_splits[-4] + ".log"
            log_file_path = os.path.join(*data_splits[:-3],log_file)

            used_zifs = []
            gather_zifs = False
            zifs_counter = 0
            with open(log_file_path, 'r') as log_file:
                for log in log_file:
                    if data_round in log or gather_zifs == True:
                        gather_zifs = True
                    else:
                        continue

                    if gather_zifs:
                        if "Selection Strategy" in log:
                            used_zifs.append(log.split(" ")[-1].replace('\n', ''))
                            zifs_counter += 1

                    if zifs_counter == key:
                        break

            threshold_dataset = pd.DataFrame()
            for zif in used_zifs:
                selected_rows = train_data.loc[train_data['type'] == zif]
                threshold_dataset = threshold_dataset._append(selected_rows, ignore_index = True)

            # Save intermediate dataset to location
            data_size_dir = os.path.join(selected_datasets_path, str(key))
            if not os.path.exists(data_size_dir):
                os.mkdir(data_size_dir)
                
            threshold_dataset.to_csv(os.path.join(data_size_dir, data_round.replace(" ","_") + "_Size_" + str(key) + ''.join(random.choice(string.ascii_letters) for _ in range(5)) +  ".csv"), index = False)

def get_datasets_by_probability(train_data: pd.DataFrame, threshold_criterion_results: dict, total_runs: int, max_dataset_size: int):
    """
    Gather the datasets that achieved a certain probability.

    Parameters:
        threshold_criterion_results (dict): A dictionary (key: data sizes, value: frequency of stopping due to threshold criterion).
        total_runs                  (int) : The total number of runs performed.

    Returns:
        None

    """   

    single_prob = False
    print("Do you want to get datasets coresponding to one probability? [Y/N] ")
    if dialogs.yesNoInput():
        single_prob = True
        print("Please enter the probability to search for.")
    else:
        print("Please enter the upper bound probability to search for.")

    probability = float(input())

    cumulative_results = cumulative_thres(threshold_criterion_results, total_runs, max_dataset_size, plot = False).copy()
    
    key_prob_distance = {}
    if single_prob:
        for key in list(cumulative_results.keys()):
            key_prob_distance[key] = abs(cumulative_results[key]["count"] - probability)
        sorted_key_prob_distance = dict(sorted(key_prob_distance.items(), key=lambda x: x[1], reverse=True))

        key_list = list(sorted_key_prob_distance.keys())[:-1]
        for key in key_list:
            del cumulative_results[key]
    else:
        for key in list(cumulative_results.keys()):
            if cumulative_results[key]["count"] > probability:
                del cumulative_results[key]

    if len(cumulative_results) == 0:
        print("There are no datasets that meet the given probability threshold.")
        return
    
    get_datasets(train_data, cumulative_results)

def get_datasets_by_data_size(train_data: pd.DataFrame, threshold_criterion_results: dict, total_runs: int, max_dataset_size: int):
    """
    Gather the datasets the achieved threshold error and are of given data size.

    Parameters:
        threshold_criterion_results (dict): A dictionary (key: data sizes, value: frequency of stopping due to threshold criterion).
        total_runs                  (int) : The total number of runs performed.
        path                        (str) : The path to the directory containing the round results.
        saveName                    (str) : The name of the directory containing the round results.

    Returns:
        None

    """   

    cumulative_results = cumulative_thres(threshold_criterion_results, total_runs, max_dataset_size, plot = False).copy()

    print("Please enter the size of the datasets you want.")
    data_size = int(input())
    while data_size not in list(cumulative_results.keys()):
        print("The data size you selected does not exist please select a number between: " + str(list(cumulative_results.keys())[0]) + " and " + str(list(cumulative_results.keys())[-1]))
        data_size = int(input())
        
    for key in list(cumulative_results.keys()):
        if key != data_size:
            del cumulative_results[key]
    
    get_datasets(train_data, cumulative_results)

def test_against_all_zifs(train_data: pd.DataFrame, full_result_thresh: dict, total_runs: int):

    # Instantiate the XGB regressor model
    XGBR = XGBRegressor(n_estimators=500, max_depth=5, eta=0.07, subsample=0.75, colsample_bytree=0.7, reg_lambda=0.4, reg_alpha=0.13,
                        n_jobs=6,
                        # nthread=6,
                        random_state=6410
                        )

    print("Do you want to select datasets by data size? [Y/N] ")
    if dialogs.yesNoInput():
        get_datasets_by_data_size(train_data, full_result_thresh, total_runs)
    else:
        get_datasets_by_probability(train_data, full_result_thresh, total_runs)
    
    for dataset_dir in os.listdir("./selected_datasets"):
        
        mae_list = []
        for dataset in os.listdir("./selected_datasets/" + dataset_dir):
            zifs, featureNames, targetNames = data_preparation(os.path.join("./selected_datasets/" + dataset_dir, dataset))

            x_trainAll = zifs[featureNames].to_numpy()
            y_trainAll = zifs[targetNames].to_numpy()

            x_test  = train_data[featureNames].to_numpy()
            y_test  = train_data[targetNames].to_numpy()

            XGBR.fit(x_trainAll, y_trainAll.ravel())

            y_pred = XGBR.predict(x_test)

            mae = metrics.mean_absolute_error(y_test, y_pred)
            mae_list.append(mae)
            print("Dataset: " + dataset + ", MAE: " + str(mae))

        print("Dataset: " + dataset_dir + ", Average MAE: " + str(sum(mae_list) / len(mae_list)))

def analyse_by_data_size(most_freq_size: int, path: str, saveName: str):
    """
    Analyzes data for a given data size.

    Parameters:
        most_freq_size (int): The most frequently encountered data size in the data.
        path           (str): The path to the directory containing the data.
        saveName       (str): The name of the directory containing the saved data.

    Returns:
        None
    """

    print("Which data size results do you want to analyse?")
    selected_size = input()

    if selected_size == "-1":
        selected_size = most_freq_size

    value_range = {}
    for experimentFile in os.listdir(path):

        # Skip non-directory files
        if not os.path.isdir(os.path.join(path, experimentFile)):
            continue

        # Skip non-directory files
        savePath = os.path.join(path, experimentFile, saveName)
        if not os.path.isdir(savePath):
            continue

        for roundDir in os.listdir(savePath):

            # Skip non-directory files
            if not os.path.isdir(os.path.join(savePath, roundDir)):
                continue                

            selectedDataset = None
            for roundResult in os.listdir(os.path.join(savePath, roundDir)):

                fileSplit = roundResult.split('_')
                # Skip the full round report.
                if fileSplit[0] != 'full':

                    if (selectedDataset is None) or (fileSplit[-2] > selectedDataset.split('_')[-2]):
                            selectedDataset = os.path.join(savePath, roundDir, roundResult)

            if selectedDataset is None:
                continue    
            dataSize = selectedDataset.split('_')[-2]
            if dataSize != selected_size:
                continue

            data = pd.read_csv(selectedDataset)
            value_range[data["mae"][1]] = {"path": selectedDataset,
                                                        "stop_criterion": data["stop_criterion"][1],
                                                        "tested_zif": data["tested_against"][1]}

    # Plot number of times each stop criterion was activated.
    thresholdCount   = 0
    performanceCount = 0
    thresholdMae   = []
    performanceMae = []
    for mae, info in value_range.items():
        if info["stop_criterion"] == "error_threshold_reached":
            thresholdCount += 1
            thresholdMae.append(mae)
        if info["stop_criterion"] == "low_performance_gain":
            performanceCount += 1
            performanceMae.append(mae)

    plt.bar(["threshold", "performance"], [thresholdCount, performanceCount], width = 0.1)
    plt.xlabel('Number of Zifs in Dataset')
    plt.ylabel('Number of times any stop criterion was met.')
    plt.show()

    if len(value_range) > 1:
        if thresholdMae:
            box_plot_statistics(thresholdMae, "Threshold Criterion")
        if performanceMae:
            box_plot_statistics(performanceMae, "No Performance Gain Criterion")
        if value_range:
            box_plot_statistics(value_range.keys(), "Range of Values")
    else:
        print("Can not further analyse data. Multitude less than 2.")

def plot_error_to_dataset_size(path :str):

    all_exp_results  = []
    all_exp_results_names = []

    point_size = 10

    more_data = True
    while more_data:

        full_exp_results = {}
        result_files_num = 0

        print("What is the label for the current data?")
        all_exp_results_names.append(input())

        for experimentFile in os.listdir(path):

            # Skip non-directory files
            if not os.path.isdir(os.path.join(path, experimentFile)):
                continue

            experimentPath = os.path.join(path, experimentFile)
            
            for content in os.listdir(experimentPath):

                if content == "bo.csv" or content == "random_opt.csv" or content == "serial_opt.csv":

                    result_files_num += 1

                    exp_data = pd.read_csv(os.path.join(experimentPath, content))

                    for index, row in exp_data.iterrows():

                        if row["sizeOfTrainingSet"] not in full_exp_results:
                            
                            full_exp_results[row["sizeOfTrainingSet"]] = {"averageError": row["averageError"] , "stdErrorOfMeanError": row["stdErrorOfMeanError"] , "stdDeviationOfMeanError": row["stdDeviationOfMeanError"]}
                        
                        else:

                            full_exp_results[row["sizeOfTrainingSet"]]["averageError"] += row["averageError"]
                            full_exp_results[row["sizeOfTrainingSet"]]["stdErrorOfMeanError"] += row["stdErrorOfMeanError"]
                            full_exp_results[row["sizeOfTrainingSet"]]["stdDeviationOfMeanError"] += row["stdDeviationOfMeanError"]

        for dataset_size in full_exp_results:
        
            full_exp_results[dataset_size]["averageError"] /= result_files_num
            full_exp_results[dataset_size]["stdErrorOfMeanError"] /= result_files_num
            full_exp_results[dataset_size]["stdDeviationOfMeanError"] /= result_files_num


        all_exp_results.append(full_exp_results)

        print("Do you want to add more data to the cumulative probability plot? [Y/N]")
        if dialogs.yesNoInput():
            print("Insert the path for the new set of data.")
            path = input()
        
        else:
            more_data = False


    if len(all_exp_results) == 1:

        exp_results = all_exp_results[0]

        results = pd.DataFrame().from_dict(exp_results, orient='index')
        results["sizeOfTrainingSet"] = results.index.astype(int)

        plot_optimization.plot_logD_trainSize_perMethod(frame1=results, label1=all_exp_results_names[0], on_off='True',
                                        size=point_size ,xLabel='Number of ZIFs in the training dataset', yLabel='Mean absolute error of logD Across Multiple Experiments',
                                        fileName=os.path.join(path, "plot_LogD-#Training_Points.png"), marker_colors=['y'])
    
    elif len(all_exp_results) == 2:

        exp_results_0 = all_exp_results[0]
        exp_results_1 = all_exp_results[1]

        results_0 = pd.DataFrame().from_dict(exp_results_0, orient='index')
        results_0["sizeOfTrainingSet"] = results_0.index.astype(int)

        results_1 = pd.DataFrame().from_dict(exp_results_1, orient='index')
        results_1["sizeOfTrainingSet"] = results_1.index.astype(int)

        stat_results = stats.ttest_rel(results_0["averageError"].to_numpy(),results_1["averageError"].to_numpy())
        print("Paired T Test Completed With p-value: " + str(stat_results.pvalue) + " and statistic: " + str(stat_results.statistic))

        result_0_v_result_1 = {"pvalue": stat_results.pvalue, "statistic": stat_results.statistic}
        print("P-Value of Paired T Test Between Method 1 and Method 2: " + str(stat_results.pvalue))
        print("Statistic Value: " + str(stat_results.statistic))

        plot_optimization.plot_logD_trainSize_perMethod(frame1=results_0, frame2=results_1, label1=all_exp_results_names[0], method1_v_method2_stats=result_0_v_result_1, label2=all_exp_results_names[1], on_off='True',
                                        size=point_size ,xLabel='Number of ZIFs in the training dataset', yLabel='Mean absolute error of logD Across Multiple Experiments',
                                        fileName=os.path.join(path, "plot_LogD-#Training_Points.png"), marker_colors=['y','g'])
        
    elif len(all_exp_results) == 3:

        exp_results_0 = all_exp_results[0]
        exp_results_1 = all_exp_results[1]
        exp_results_2 = all_exp_results[2]

        results_0 = pd.DataFrame().from_dict(exp_results_0, orient='index')
        results_0["sizeOfTrainingSet"] = results_0.index.astype(int)

        results_1 = pd.DataFrame().from_dict(exp_results_1, orient='index')
        results_1["sizeOfTrainingSet"] = results_1.index.astype(int)

        results_2 = pd.DataFrame().from_dict(exp_results_2, orient='index')
        results_2["sizeOfTrainingSet"] = results_2.index.astype(int)

        stat_results = stats.ttest_rel(results_0["averageError"].to_numpy(),results_1["averageError"].to_numpy())
        print("Paired T Test Completed With p-value: " + str(stat_results.pvalue) + " and statistic: " + str(stat_results.statistic))

        result_0_v_result_1 = {"pvalue": stat_results.pvalue, "statistic": stat_results.statistic}
        print("P-Value of Paired T Test Between Method 1 and Method 2: " + str(stat_results.pvalue))
        print("Statistic Value: " + str(stat_results.statistic))

        stat_results = stats.ttest_rel(results_0["averageError"].to_numpy(),results_2["averageError"].to_numpy())
        print("Paired T Test Completed With p-value: " + str(stat_results.pvalue) + " and statistic: " + str(stat_results.statistic))

        result_0_v_result_2 = {"pvalue": stat_results.pvalue, "statistic": stat_results.statistic}
        print("P-Value of Paired T Test Between Method 1 and Method 3: " + str(stat_results.pvalue))
        print("Statistic Value: " + str(stat_results.statistic))

        plot_optimization.plot_logD_trainSize_perMethod(frame1=results_0, frame2=results_1, frame3=results_2, method1_v_method2_stats=result_0_v_result_1, method1_v_method3_stats=result_0_v_result_2 ,label1=all_exp_results_names[0], label2=all_exp_results_names[1], label3=all_exp_results_names[2], on_off='True',
                                        size=point_size ,xLabel='Number of ZIFs in the training dataset', yLabel='Mean absolute error of logD Across Multiple Experiments',
                                        fileName=os.path.join(path, "plot_LogD-#Training_Points.png"), marker_colors=['y','g','r'])
    
    else:
        print("Method is implemented for up to three types of experiments.")