#! /bin/env python
# Import library
from ga_inverse import *
import argparse
import sys

# Command line parameters
parser = argparse.ArgumentParser(
                    prog='ga_inverse_testlib_non-jupyter',
                    description='This script: (a) creates an ML model connecting MOF and gas parameters to an expected functional output (logDi & logDi/logDj, for species i and j), ' + 
                        '(b) given a target functional output in the specific (MOF,gas) setting, searches and suggests promising MOF design parameters.',
                    epilog='For more information check the paper at: https://chemrxiv.org/engage/chemrxiv/article-details/642c2e0a16782ec9e6557a3e')

parser.add_argument('-c', '--case', choices=['co2','propylene', 'o2'], help='The gas used in the setting.', default='co2')
parser.add_argument('-t', '--trainingDataFile', help='The (XLSX) datafile containing the training data.', default='./TrainData.xlsx')
parser.add_argument('-lD', '--lowerD', help='TODO', type=float)
parser.add_argument('-hD', '--higherD', help='TODO',type=float)
parser.add_argument('-lR', '--lowerR', help='TODO',type=float)
parser.add_argument('-hR', '--higherR', help='TODO',type=float)
parser.add_argument('-r', '--rounds', help='TODO',type=int, default=1)
parser.add_argument('-g', '--generations', help='TODO',type=int, default=100)
parsed_args = parser.parse_args() # Actually parse
print("Using parameters:\n%s"%(str(parsed_args)))


selected_case = parsed_args.case # You can choose among 'propylene', 'o2' and 'co2', which correspond to propylene/propane, o2/n2 and co2/ch4 mixtures, respectively
separation = selected_case
diameter_tuple, mass_tuple, ascF_tuple, kD_tuple, linker_length1, func1_length, metalNum, linker_length3, func3_length, GeneFieldNames, gene_space = case(selected_case)
num_generations = parsed_args.generations

# Read data
data_from_file = readData(parsed_args.trainingDataFile)
training_data, gene_repr_of_training_data, training_x, training_y = prepareDataForLearning(data_from_file, GeneFieldNames)


# Train model
model = train_model(training_x, training_y)


# TODO: Time the process
# TODO: Keep the top N 

# For every possible combination of the input parameters, get the expected logD of gas_i
for cur_linker_length1 in linker_length1.keys():
    for cur_func1_length in func1_length.keys():
        for cur_linker_length2 in linker_length2.keys():
            for cur_func2_length in func2_length.keys():
                for cur_linker_length3 in linker_length3.keys():
                    for cur_func3_length in func3_length.keys():
                        for metalNum in metalNum.keys():            
                            # Here get all the values
                            cur_linker_length1_values = linker_length1[cur_linker_length1]
                            linker_mass1 = linker_mass1
                            sigma1 = linker_mass1['σ1']
                            # Etc.
                            # Make sure you reuse linker1 info for linker2 and/or linker3 info as needed
                            if linker_length3 is None:
                                pass # TODO 

                            # Linker 2 and func2 always use the linker1, func1 data
                            pass # TODO

                            # Same for func3
                            if func3_length is None:
                                pass # TODO


                            # Get the model output
                            # Print/save the result
