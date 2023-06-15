#! /bin/env python
# Import library
from ga_inverse import *
import argparse

# Command line parameters
parser = argparse.ArgumentParser(
                    prog='ga_inverse_testlib_non-jupyter',
                    description='This script: (a) creates an ML model connecting MOF and gas parameters to an expected functional output (logDi & logDi/logDj, for species i and j), ' + 
                        '(b) given a target functional output in the specific (MOF,gas) setting, searches and suggests promising MOF design parameters.',
                    epilog='For more information check the paper at: https://chemrxiv.org/engage/chemrxiv/article-details/642c2e0a16782ec9e6557a3e')

parser.add_argument('-c', '--case', choices=['co2','propylene', 'o2'], help='The gas used in the setting.', default='co2')
parser.add_argument('-t', '--trainingDataFile', help='The (XLSX) datafile containing the training data.', default='./TrainData.xlsx')
parsed_args = parser.parse_args() # Actually parse
print("Using parameters:\n%s"%(str(parsed_args)))


selected_case = parsed_args.case # You can choose among 'propylene', 'o2' and 'co2', which correspond to propylene/propane, o2/n2 and co2/ch4 mixtures, respectively
diameter_tuple, mass_tuple, ascF_tuple, kD_tuple, linker_length1, func1_length, metalNum, linker_length3, func3_length, GeneFieldNames, gene_space = case(selected_case)

# Read data
data_from_file = readData(parsed_args.trainingDataFile)
training_data, gene_repr_of_training_data, training_x, training_y = prepareDataForLearning(data_from_file, GeneFieldNames)


# Train model
model = train_model(training_x, training_y)


# TODO: Time the process
# TODO: Keep the top N 

# For every possible combination of the input parameters, get the expected logD of gas_i
# TODO: implement with iterators?

# Common fields across cases
linker_length1_keys = gene_space[1]
# Differentiated fields per case
if selected_case == 'o2':
     metalNum_keys = gene_space[0]
     linker_length3_keys = gene_space[2]
     func1_length_keys = gene_space[3]
     func3_length_keys = gene_space[4]
elif selected_case == 'co2':
     metalNum_keys = gene_space[0]
     func1_length_keys = gene_space[2]
elif selected_case == 'propylene':
     metalNum_keys = gene_space[0]
     func1_length_keys = gene_space[2]
else:
     raise RuntimeError("Invalid case type (%s). Aborting..."%(selected_case))

# Update length3 and func3, if needed
if linker_length3 is None:
        linker_length3_keys = linker_length1_keys
if func3_length is None:
    func3_length_keys = func1_length_keys

solution_index = 0 # Init counter
results = [] # Init results list

for cur_linker_length1 in linker_length1_keys: #
    for cur_func1_length in func1_length_keys:            
        for cur_linker_length3 in linker_length3_keys:
            for cur_func3_length in func3_length_keys:
                for cur_metal_num in metalNum_keys:
                    solution_index +=1  # Update counter

                    # Solution format (from library comments)
                    # if ((separation == 'propylene') or (separation == 'co2')):
                    #         gene_field_names = ['MetalNum','linker_length1','func1_length']
                    #     else:
                    #         gene_field_names = ['MetalNum','linker_length1','linker_length3', 'func1_length',
                    #                                 'func3_length']                    
                    if ((selected_case == 'propylene') or (selected_case == 'co2')):
                        solution = [cur_metal_num, cur_linker_length1, cur_func1_length]
                    elif (selected_case == 'o2'):
                        solution = [cur_metal_num, cur_linker_length1, cur_linker_length3, cur_func1_length,cur_func3_length]
                    else:
                        raise RuntimeError('Invalid case (%s). Aborting...'%(selected_case))
                            
                    # Get the model output
                    estimated_gas1_diffusivity, estimated_gas2_diffusivity = estimate_diffusivities_from_solution(solution, 
                                                                selected_case, diameter_tuple, mass_tuple, ascF_tuple, 
                                                                kD_tuple, metalNum, model, linker_length1, func1_length, 
                                                                linker_length3=linker_length3, func3_length=func3_length
                                                                )
                    ratio = estimated_gas1_diffusivity - estimated_gas2_diffusivity
                    row = [*solution, estimated_gas1_diffusivity, estimated_gas2_diffusivity, ratio]
                    print(str(row)) # Output
                    results.append(row) # Append to results list
# Print/save the result
results_as_matrix = np.array(results)
np.savetxt("results.csv", results_as_matrix, delimiter=",")

# NOTES:
                    # # Here get all the values
                    # cur_linker_length1_values = linker_length1[cur_linker_length1]
                    # cur_linker_length3_values = linker_length3[cur_linker_length3]
                    # metalNumValues = MetalNum[curMetalNum]

                    # # Linker 2 and func2 always use the linker1, func1 data
                    # cur_linker_length2 = cur_linker_length1
                    # cur_func2_length = cur_func1_length

                    # # Get individual fields
                    # # {'linker_mass1':83, 
                    # linker_mass1 = cur_linker_length1_values['linker_mass1']
                    # # 'σ_1':0.325, 
                    # sigma1 = cur_linker_length1_values['σ1']
                    # # 'e_1':0.7112, 
                    # epsilon1 = cur_linker_length1_values['e1']
                    # # 'linker_length2':3.66, 
                    # linker_length2 = cur_linker_length1_values['linker_length2']
                    # # 'linker_mass2':83, 
                    # linker_mass2 = cur_linker_length1_values['linker_mass2']
                    # # 'σ_2':0.325, 
                    # sigma2 = cur_linker_length1_values['σ2']
                    # # 'e_2':0.7112, 
                    # epsilon2 = cur_linker_length1_values['e2']
                    # # 'linker_length3':3.66,
                    # linker_length3 = cur_linker_length3_values['linker_length3']
                    # # 'linker_mass3':83, 
                    # linker_mass3 = cur_linker_length3_values['linker_mass3']
                    # # 'σ_3':0.325, 
                    # sigma3 = cur_linker_length3_values['σ3']
                    # # 'e_3':0.7112}
                    # epsilon3 = cur_linker_length1_values['e3']

                    # # MetalNum
                    # # {'ionicRad':41,
                    # ionicRad = metalNumValues['ionicRad']
                    # #  'MetalMass': 9.012}                       
                    # MetalMass = metalNumValues['MetalMass']

