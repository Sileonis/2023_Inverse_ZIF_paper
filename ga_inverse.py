# Main imports
import pandas as pd
import numpy as np

# Type hinting
from typing import Callable

# Learning
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar
from xgboost import XGBRegressor

# Data splitting
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold


# Plotting and visualization
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import StrMethodFormatter
from tabulate import _table_formats, tabulate

# Indexing
# CURRENTLY Unimportant
# from rtree import index

# Search / optimization
import pygad

# Utility
import time
import datetime


#####
#function to define the separation problem case
#####
def case(separation):
    if separation == 'propylene':
        diameter_tuple = (np.array([4.03]), np.array([4.16]))
        mass_tuple = (np.array([42.08]), np.array([44.1]))
        ascF_tuple = (np.array([0.142]), np.array([0.152]))
        kD_tuple = (np.array([4.5]), np.array([4.3]))
    elif separation == 'co2':
        diameter_tuple = (np.array([3.24]), np.array([3.25]))
        mass_tuple = (np.array([44.01]), np.array([16.04]))
        ascF_tuple = (np.array([0.225]), np.array([0.011]))
        kD_tuple = (np.array([3.3]), np.array([3.8]))
    elif separation == 'o2':
        diameter_tuple = (np.array([2.94]), np.array([3.13]))
        mass_tuple = (np.array([31.999]), np.array([28.000]))
        ascF_tuple = (np.array([0.022]), np.array([0.037]))
        kD_tuple = (np.array([3.3]), np.array([3.8]))
    
    if separation == 'propylene' or separation=='co2':
        linker_length1 = {3.66:{'linker_mass1':83, 'σ_1':0.325, 'e_1':0.7112, 'linker_length2':3.66, 'linker_mass2':83, 'σ_2':0.325, 'e_2':0.7112, 'linker_length3':3.66, 'linker_mass3':83, 'σ_3':0.325, 'e_3':0.7112},
                 4.438:{'linker_mass1':81, 'σ_1':0.25, 'e_1':0.0627, 'linker_length2':4.438, 'linker_mass2':81, 'σ_2':0.25, 'e_2':0.0627, 'linker_length3':4.438, 'linker_mass3':81, 'σ_3':0.25, 'e_3':0.0627},
                  4.86:{'linker_mass1':101.98, 'σ_1':0.285, 'e_1':0.255, 'linker_length2':4.86, 'linker_mass2':101.98, 'σ_2':0.285, 'e_2':0.255, 'linker_length3':4.86, 'linker_mass3':101.98, 'σ_3':0.285, 'e_3':0.255},
                   5.7:{'linker_mass1':134.906, 'σ_1':0.34, 'e_1':1.2552, 'linker_length2':5.7, 'linker_mass2':134.906, 'σ_2':0.34, 'e_2':1.2552, 'linker_length3':5.7, 'linker_mass3':134.906, 'σ_3':0.34, 'e_3':1.2552},
                  6.01:{'linker_mass1':223.8, 'σ_1':0.4, 'e_1':1.8731, 'linker_length2':6.01, 'linker_mass2':223.8, 'σ_2':0.4, 'e_2':1.8731, 'linker_length3':6.01, 'linker_mass3':223.8, 'σ_3':0.4, 'e_3':1.8731},
                  6.41:{'linker_mass1':317.8, 'σ_1':0.367, 'e_1':2.4267, 'linker_length2':6.41, 'linker_mass2':317.8, 'σ_2':0.367, 'e_2':2.4267, 'linker_length3':6.01, 'linker_mass3':223.8, 'σ_3':0.367, 'e_3':2.4267}
                  }
        
        func1_length = {2.278:{'func1_mass':1., 'func2_length': 2.278, 'func2_mass':1., 'func3_length': 2.278, 'func3_mass':1.},
                 3.54:{'func1_mass':35.45, 'func2_length': 3.54, 'func2_mass':35.45, 'func3_length': 3.54, 'func3_mass':35.45},
                  3.78:{'func1_mass':15., 'func2_length': 3.78, 'func2_mass':15., 'func3_length': 3.78, 'func3_mass':15.},
                   3.85:{'func1_mass':79.9, 'func2_length': 3.85, 'func2_mass':79.9, 'func3_length': 3.85, 'func3_mass':79.9},
                  3.927:{'func1_mass':16., 'func2_length': 3.927, 'func2_mass':16., 'func3_length': 3.927, 'func3_mass':16.},
                  4.093:{'func1_mass':31., 'func2_length': 4.093, 'func2_mass':31., 'func3_length': 4.093, 'func3_mass':31.}
                  }

        linker_length3 = None
        func3_length = None
        
        MetalNum = {4:{'ionicRad':41, 'MetalMass': 9.012},
           29:{'ionicRad':71,'MetalMass': 63.456},
           12:{'ionicRad':71, 'MetalMass': 24.305},
           27:{'ionicRad':72, 'MetalMass': 58.930},
           30:{'ionicRad':74, 'MetalMass': 65.380},
           25:{'ionicRad':80, 'MetalMass': 54.938},
           48:{'ionicRad':92, 'MetalMass': 112.411}}

        # TODO: Fix
        gene_space = [
                [ 4,12,25,27,29,30,48],
                [ 4.438, 4.86, 5.7, 6.01, 6.41],
                [ 3.54, 3.78, 3.85,  4.093],
            ]
        
    elif separation == 'o2':
        linker_length1 = {3.66:{'linker_mass1':83, 'σ_1':0.325, 'e_1':0.7112, 'linker_length2':3.66, 'linker_mass2':83, 'σ_2':0.325, 'e_2':0.7112},
                 4.438:{'linker_mass1':81, 'σ_1':0.25, 'e_1':0.0627, 'linker_length2':4.438, 'linker_mass2':81, 'σ_2':0.25, 'e_2':0.0627},
                  4.86:{'linker_mass1':101.98, 'σ_1':0.285, 'e_1':0.255, 'linker_length2':4.86, 'linker_mass2':101.98, 'σ_2':0.285, 'e_2':0.255},
                   5.7:{'linker_mass1':134.906, 'σ_1':0.34, 'e_1':1.2552, 'linker_length2':5.7, 'linker_mass2':134.906, 'σ_2':0.34, 'e_2':1.2552},
                  6.01:{'linker_mass1':223.8, 'σ_1':0.4, 'e_1':1.8731, 'linker_length2':6.01, 'linker_mass2':223.8, 'σ_2':0.4, 'e_2':1.8731},
                  6.41:{'linker_mass1':317.8, 'σ_1':0.367, 'e_1':2.4267, 'linker_length2':6.41, 'linker_mass2':317.8, 'σ_2':0.367, 'e_2':2.4267}
                  }

        linker_length3 = {3.66:{'linker_mass3':83, 'σ_3':0.325, 'e_3':0.7112},
                        4.438:{'linker_mass3':81, 'σ_3':0.25, 'e_3':0.0627},
                        4.86:{'linker_mass3':101.98, 'σ_3':0.285, 'e_3':0.255},
                        5.7:{'linker_mass3':134.906, 'σ_3':0.34, 'e_3':1.2552},
                        5.996:{'linker_mass3':117., 'σ_3':0.25, 'e_3':0.0627},
                        6.01:{'linker_mass3':223.8, 'σ_3':0.4, 'e_3':1.8731},
                        6.41:{'linker_mass3':317.8, 'σ_3':0.367, 'e_3':2.4267}
                        }

        func1_length = {2.278:{'func1_mass':1., 'func2_length': 2.278, 'func2_mass':1.},
                        3.54:{'func1_mass':35.45, 'func2_length': 3.54, 'func2_mass':35.45},
                        3.78:{'func1_mass':15., 'func2_length': 3.78, 'func2_mass':15.},
                        3.85:{'func1_mass':79.9, 'func2_length': 3.85, 'func2_mass':79.9},
                        3.927:{'func1_mass':16., 'func2_length': 3.927, 'func2_mass':16.},
                        4.093:{'func1_mass':31., 'func2_length': 4.093, 'func2_mass':31.}
                        }

        func3_length = {2.278:{'func3_mass':1.},
                            2.7:{'func3_mass':18.99},
                        3.54:{'func3_mass':35.45},
                        3.78:{'func3_mass':15.},
                        3.85:{'func3_mass':79.9},
                        3.927:{'func3_mass':16.},
                        4.093:{'func3_mass':31.},
                        4.25:{'func3_mass':127.},
                        }

        MetalNum = {4:{'ionicRad':41, 'MetalMass': 9.012},
                29:{'ionicRad':71,'MetalMass': 63.456},
                12:{'ionicRad':71, 'MetalMass': 24.305},
                27:{'ionicRad':72, 'MetalMass': 58.930},
                30:{'ionicRad':74, 'MetalMass': 65.380},
                25:{'ionicRad':80, 'MetalMass': 54.938},
                48:{'ionicRad':92, 'MetalMass': 112.411}}

        # TODO: Fix
        gene_space =  [
                    [ 4,12,25,27,29,30,48],
                    [ 4.438, 4.86, 5.7, 6.01, 6.41],
                    [ 4.438, 4.86, 5.7,5.996,6.01, 6.41],
                    [ 3.54, 3.78, 3.85,  4.093],
                    [2.278,2.7,3.54, 3.78, 3.85,  4.093,4.25]
                    ]
        
    if ((separation == 'propylene') or (separation == 'co2')):
        gene_field_names = ['MetalNum','linker_length1','func1_length']
    else:
        gene_field_names = ['MetalNum','linker_length1','linker_length3', 'func1_length',
                                   'func3_length']
    return diameter_tuple, mass_tuple, ascF_tuple, kD_tuple, linker_length1, func1_length, MetalNum, linker_length3, func3_length, gene_field_names, gene_space


def readData(source_file = './TrainData.xlsx') -> pd.DataFrame:
    # Read file
    df=pd.read_excel(source_file)
    df.head(2)
    df['logD'] = np.log10(df['diffusivity'])

    # Keep appropriate columns
    cleaned_original_df=df[[ 'type', 'gas', 'MetalNum', 'aperture', 'size - van der Waals (Å)','mass', 'ascentricF', 'logD', 'size - kinetic diameter (Å)', 'ionicRad', 
        'Μ-N_lff', 'Μ-N_kFF', 'MetalCharge', 'MetalMass',
        'σ_1', 'e_1', 'σ_2', 'e_2', 'σ_3', 'e_3', 'linker_length1', 'linker_length2',
        'linker_length3', 'linker_mass1', 'linker_mass2', 'linker_mass3',
        'func1_length', 'func2_length', 'func3_length', 'func1_mass',  
        'func2_mass', 'func3_mass', 'func1_charge', 'func2_charge',
        'func3_charge',]]
    
    # Rename columns
    cleaned_original_df=cleaned_original_df.rename(columns={'size - van der Waals (Å)':'diameter', 'size - kinetic diameter (Å)':'kdiameter', 
        })

    # Clear NA entries
    cleaned_original_df = cleaned_original_df.dropna()
    # Remove outlier molecule (?)    
    cleaned_original_df=cleaned_original_df.reset_index(drop=True)
   
    return cleaned_original_df

def prepareDataForLearning(dataframe, gene_field_names  ) -> tuple:
    """Given an appropriate input dataframe of training data, returns a tuple containing: 
     - the dataframe with the appropriate training columns, 
     - the gene representations of the input data, 
     - the representation of the training data inlcuding all dependent and independent features,
     - the target values for each instance of the training data. 
    """
    selected_cols_df = dataframe[[
     'MetalNum',  
     'linker_length1', 'linker_length2', 'linker_length3',
    'func1_length', 'func2_length', 'func3_length' 
                                 ]]

    selected_cols_df = selected_cols_df.drop_duplicates()
    selected_cols_df = selected_cols_df.reset_index(drop=True)

    Genes = np.asanyarray(selected_cols_df[gene_field_names])

    x_all = np.asanyarray(dataframe[[
        'diameter',
        'mass',
        'ascentricF',
        'kdiameter',
        'MetalNum',  
        'linker_length1', 'linker_length3',
        'func1_length',  'func3_length',
        'linker_length2',
        'func2_length', 
        'ionicRad', 'MetalMass',
        'linker_mass1', 'linker_mass2', 'linker_mass3',
        'σ_1', 'e_1', 'σ_2', 'e_2', 'σ_3', 'e_3', 
        'func1_mass', 'func2_mass', 'func3_mass'
                                ]])

    y_all = np.array(dataframe[['logD']])    

    return (selected_cols_df, Genes,x_all, y_all)

def train_model(x_all, y_all):
    XGBR = XGBRegressor(n_estimators=800, max_depth=5, eta=0.02, subsample=0.75, colsample_bytree=0.3, reg_lambda=0.6, reg_alpha=0.15,
                    # n_jobs=6,
                    nthread=1,
                    random_state=61
                   )

    model = XGBR.fit(x_all, y_all)

    return model

def get_base_vector_from_solution(solution, separation, linker_length1, func1_length, metalNum, linker_length3 = None, func3_length = None):
    if ((separation == 'propylene') or (separation == 'co2')):
        base_vector = np.array([
                  linker_length1[solution[1]]['linker_length2'],
                  linker_length1[solution[1]]['linker_length3'],
                  func1_length[solution[2]]['func2_length'],
                  func1_length[solution[2]]['func3_length'],
                  metalNum[solution[0]]['ionicRad'],
                  metalNum[solution[0]]['MetalMass'],
                  linker_length1[solution[1]]['linker_mass1'],
                  linker_length1[solution[1]]['linker_mass2'],
                  linker_length1[solution[1]]['linker_mass3'],
                  linker_length1[solution[1]]['σ_1'],
                  linker_length1[solution[1]]['e_1'],
                  linker_length1[solution[1]]['σ_2'],
                  linker_length1[solution[1]]['e_2'],
                  linker_length1[solution[1]]['σ_3'],
                  linker_length1[solution[1]]['e_3'],
                  func1_length[solution[2]]['func1_mass'],
                  func1_length[solution[2]]['func2_mass'],
                  func1_length[solution[2]]['func3_mass']])
    elif (separation == 'o2'):
        base_vector=np.array([
                  linker_length1[solution[1]]['linker_length2'],
                  func1_length[solution[3]]['func2_length'],
                  metalNum[solution[0]]['ionicRad'],
                  metalNum[solution[0]]['MetalMass'],
                  linker_length1[solution[1]]['linker_mass1'],
                  linker_length1[solution[1]]['linker_mass2'],
                  linker_length3[solution[2]]['linker_mass3'],
                  linker_length1[solution[1]]['σ_1'],
                  linker_length1[solution[1]]['e_1'],
                  linker_length1[solution[1]]['σ_2'],
                  linker_length1[solution[1]]['e_2'],
                  linker_length3[solution[2]]['σ_3'],
                  linker_length3[solution[2]]['e_3'],
                  func1_length[solution[3]]['func1_mass'],
                  func1_length[solution[3]]['func2_mass'],
                  func3_length[solution[4]]['func3_mass']])

    return base_vector

def get_whole_vector_per_gas_from_solution(solution, separation, diameter_tuple, mass_tuple, ascF_tuple, kD_tuple,
        metalNum, 
         linker_length1, func1_length, linker_length3 = None, func3_length = None):
    diameter_gas1, diameter_gas2 = diameter_tuple
    mass_gas1, mass_gas2 = mass_tuple
    ascF_gas1, ascF_gas2 = ascF_tuple
    kD_gas1, kD_gas2 = kD_tuple

    solution2=get_base_vector_from_solution(solution, separation, linker_length1, func1_length, 
                                            metalNum,
                                            linker_length3, func3_length)
    solution_gas1=np.concatenate((diameter_gas1, mass_gas1, ascF_gas1, kD_gas1, solution,solution2), axis=0, out=None, dtype=None, casting="same_kind")
    solution_gas2=np.concatenate((diameter_gas2, mass_gas2, ascF_gas2, kD_gas2,solution,solution2), axis=0, out=None, dtype=None, casting="same_kind")

    return solution_gas1, solution_gas2

def estimate_diffusivities_from_solution(solution, separation, diameter_tuple, mass_tuple, ascF_tuple, kD_tuple, 
                                         metalNum, model,
                                        linker_length1, func1_length,
                                        linker_length3 = None, func3_length = None):
    solution_gas1, solution_gas2 = get_whole_vector_per_gas_from_solution(solution, separation, 
                                        diameter_tuple, mass_tuple, ascF_tuple, kD_tuple,
                                        metalNum, 
                                        linker_length1=linker_length1, func1_length=func1_length,
                                        linker_length3=linker_length3, func3_length=func3_length)
    
    estimated_gas1_diffusivity  = model.predict([solution_gas1])[0]
    estimated_gas2_diffusivity  = model.predict([solution_gas2])[0]

    return estimated_gas1_diffusivity, estimated_gas2_diffusivity

def fitness_base(solution, solution_idx, separation, diameter_tuple, mass_tuple, ascF_tuple, kD_tuple, 
                 boundaries_D, boundaries_R, linker_length1, func1_length,
                 linker_length3 = None, func3_length = None,
                 model = None, customFitnessFormula : Callable[[float, float], float] = None
                ) -> float:
    
    if model is None:
        raise RuntimeError("Model needs to be specified. Aborting.")
    
    estimated_gas1_diffusivity, estimated_gas2_diffusivity = estimate_diffusivities_from_solution(solution, 
                                                                separation, diameter_tuple, mass_tuple, ascF_tuple, 
                                                                kD_tuple, model, linker_length1, func1_length, 
                                                                linker_length3=linker_length3, func3_length=func3_length
                                                                )

    # If no fitness is defined
    if (customFitnessFormula == None):
        Ratio = estimated_gas1_diffusivity - estimated_gas2_diffusivity
        
        DiffusivityContribution = 1.0/(abs(np.min(boundaries_D) - estimated_gas1_diffusivity) + abs(np.max(boundaries_D) - estimated_gas1_diffusivity))
        RatiosContribution = 1.0/(abs(np.min(boundaries_R) - Ratio) + abs(np.max(boundaries_R) - Ratio))
        
        overallFitnessMeasure = 0.5*DiffusivityContribution + 0.5*RatiosContribution
    else:
        overallFitnessMeasure = customFitnessFormula(estimated_gas1_diffusivity, estimated_gas2_diffusivity)
    
    return overallFitnessMeasure


#############
# Field-related constants
GENE_FIELDS = [
        'MetalNum',  
        'linker_length1', 'linker_length2', 'linker_length3',
        'func1_length', 'func2_length', 'func3_length' 
                                    ]
#############

def represent_instances_as_genes(instances_dataframe: pd.DataFrame) -> np.array:
    return np.asanyarray(instances_dataframe[GENE_FIELDS])


def prepareGA(fitness, starting_population_data, gene_space, suppress_warnings=False, **kwargs):
    fitness_function = fitness

    # TODO: Replace all parameters with default values from pygad initializer
    if 'random_seed' not in kwargs:
        random_seed=None
    else:
        random_seed = kwargs['random_seed']

    if 'num_generations' not in kwargs:
        num_generations = 100
    else:
        num_generations = kwargs['num_generations']
    
    if 'num_parents_mating' not in kwargs:
        num_parents_mating = 6 # 20,  22  #number of solutions to be selected as parents    
    else:
        num_parents_mating = kwargs['num_parents_mating']
    

    if 'mutation_probability' not in kwargs:
        mutation_probability = [0.6, 0.03] # originally [0.4, 0.01]
    else:
        mutation_probability = kwargs['mutation_probability']

    if 'mutation_type' not in kwargs:
        mutation_type = "adaptive" # random, swap, inversion, scramble, adaptive
    else:
        mutation_type = kwargs['mutation_type']

    Genes = np.asanyarray(starting_population_data)

    initial_population = Genes

    # narrowed down options
    # if 'gene_space' not in kwargs:
    #     print("[WARNING]: no gene space defined... Using a default gene space.")
    #     gene_space =  [
    #                 [ 4,12,25,27,29,30,48],
    #                 [ 4.438, 4.86, 5.7, 6.01, 6.41],
    #                 [ 4.438, 4.86, 5.7, 6.01, 6.41],
    #                 [ 4.438, 4.86, 5.7,5.996,6.01, 6.41],
    #                 [ 3.54, 3.78, 3.85,  4.093],
    #                 [ 3.54, 3.78, 3.85,  4.093],
    #                 # [ 3.54, 3.78, 3.85,  4.093]
    #                 [2.278, 2.7, 3.54, 3.78, 3.85,  4.093, 4.25],
    #                 ]
    # else:
    #     gene_space = kwargs['gene_space']

    if 'parent_selection_type' not in kwargs:   
        parent_selection_type = "tournament" #  "sss", "rws", "tournament", "rank", "random"
    else:
        parent_selection_type = kwargs['parent_selection_type']
    
    if 'K_tournament' not in kwargs:
        K_tournament = 6 #12
    else:
        K_tournament = kwargs['K_tournament']
    

    if 'keep_parents' not in kwargs:
        keep_parents = 2
    else:
        keep_parents = kwargs['keep_parents']
        
    if 'keep_elitism' not in kwargs:
        keep_elitism = 2
    else:
        keep_elitism = kwargs['keep_elitism']

    if 'crossover_type' not in kwargs:
        crossover_type = "uniform" # single_point, two_points, uniform, scattered
    else:
        crossover_type = kwargs['crossover_type']

    if 'crossover_probability' not in kwargs:
        crossover_probability = 0.7
    else:
        crossover_probability = kwargs['crossover_probability']
    
    if 'stop_criteria' not in kwargs:
        stop_criteria=["saturate_100"]
    else:
        stop_criteria = kwargs['stop_criteria']
    
    if 'on_generation' not in kwargs:
        def on_generation_func(ga):
            print("Generation", ga.generations_completed)
            print(ga.population)
        on_generation = on_generation_func
    else:
        on_generation = kwargs['on_generation']
    
    # Actually initialize GA
    ga_instance = pygad.GA(num_generations=num_generations,
                        num_parents_mating=num_parents_mating,
                        fitness_func=fitness_function,
                        initial_population = initial_population, 
                        parent_selection_type=parent_selection_type,
                        keep_elitism=keep_elitism,
                        crossover_type=crossover_type,
                        mutation_type=mutation_type, 
                        gene_space = gene_space,
                        crossover_probability = crossover_probability,
                        mutation_probability = mutation_probability,
                        K_tournament = K_tournament,
                        allow_duplicate_genes=False,
                        save_solutions=True,
                        random_seed=None,
                        suppress_warnings=True,
                        parallel_processing= 4,
                        save_best_solutions=True
                      )
        
    return ga_instance

def runGA(ga_instance):
    """Returns the solution (as a gene), the related fitness and its index."""

    now = datetime.datetime.now()
    print ("Current date and time : ")
    print (now.strftime("%Y-%m-%d %H:%M:%S"))
    start_time = time.time()

    # Actually run the GA
    ga_instance.run()

    print("--- %s minutes ---" % ((time.time() - start_time)/60))
    now = datetime.datetime.now()
    print ("Current date and time : ")
    print (now.strftime("%Y-%m-%d %H:%M:%S"))

    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print("Parameters of the best solution : {solution}".format(solution=solution))
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

    # Obsolete
    # solution2 = get_base_vector_from_solution(solution)

    return solution, solution_fitness, solution_idx

def evaluate_solution(solution, separation, model, diameter_tuple, mass_tuple, ascF_tuple, kD_tuple,
                        metalNum, 
                        linker_length1, func1_length,
                        linker_length3=None, func3_length=None):
    solution_gas1, solution_gas2= get_whole_vector_per_gas_from_solution(solution, separation,
                                                                         diameter_tuple, 
                                                                         mass_tuple, ascF_tuple, kD_tuple,
                                                                         metalNum, 
                                                                         linker_length1, func1_length,
                                                                         linker_length3, func3_length
                                                                         )

    prediction_gas1 = model.predict([solution_gas1])[0]
    prediction_gas2 = model.predict([solution_gas2])[0]

    print(prediction_gas1)
    print(prediction_gas2)
    S= prediction_gas1 - prediction_gas2
    print(S)
    
    return S

def get_best_solutions(ga_instance):
    dfSol=pd.DataFrame(data = ga_instance.best_solutions)

    dfSol2=dfSol.drop_duplicates(keep='last')
    dfSol3 = dfSol2.reset_index()
    dfSol3 = dfSol3.drop(columns=['index'])

    return np.asanyarray(dfSol3) # TODO: Check

def plot_results(ga_instance, solution_rate=True, fitness_per_generation = True, gene_plot=True):
    if solution_rate:
        # New solution rate
        print(ga_instance.plot_new_solution_rate())    

    if fitness_per_generation:
        # Fitness per generation
        print(ga_instance.plot_fitness(plot_type="scatter"))

    if gene_plot:
        # Gene plot
        print(ga_instance.plot_genes(graph_type="plot",
                        plot_type="scatter",
                        solutions='all'))


def plot_logDvsRatio(dataForPlotting, gas1_name, gas2_name, logD_field_name = 'logD', ratio_field_name = 'Ratio', 
                     save_to_file = False, filename = None):
    plt.plot(dataForPlotting[logD_field_name], dataForPlotting[ratio_field_name], 's', label='GA ZIFs', c='b', markersize='12', markeredgewidth=2, markeredgecolor='k')
    plt.xlabel('log$D_{%s}$'%(gas1_name), fontsize=20)
    plt.ylabel('selectivity (log($D_{%s}$ / $D_{%s}$)'%(gas1_name, gas2_name),fontsize=20)
    plt.legend(loc='lower right', fontsize=20)


    plt.tick_params(which='both', width=3)
    plt.tick_params(which='major', length=10)
    plt.rcParams["figure.figsize"] = (6,6)
    plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))

    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)

    # If asked to save
    if save_to_file:
        # If we do not have a filename
        if filename is None:
            # Create one
            filename = 'performance_%s_%s.png'%(gas1_name, gas2_name)
        plt.savefig(filename, bbox_inches='tight', dpi=300, transparent=True)
    
    plt.show()

def plot_fitnessPerMOF(dataForPlotting, gas1_name, gas2_name, id_field_name = 'MOF_ID', fitness_field_name = 'fitness',
                        save_to_file = False, filename = None):
    plt.plot(dataForPlotting[id_field_name], dataForPlotting[fitness_field_name], 's', label='GA ZIFs', c='b', markersize='18', markeredgewidth=1, markeredgecolor='k')
    plt.xlabel('# of candidate', fontsize=20)
    plt.ylabel('fitness',fontsize=20)
    plt.legend(loc='lower right', fontsize=20)


    plt.tick_params(which='both', width=3)
    plt.tick_params(which='major', length=10)
    plt.rcParams["figure.figsize"] = (6,6)
    plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))

    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)

    # If asked to save
    if save_to_file:
        # If we do not have a filename
        if filename is None:
            # Create one
            filename = 'fitness_%s_%s.png'%(gas1_name, gas2_name)
        plt.savefig(filename, bbox_inches='tight', dpi=300, transparent=True)
    plt.show()

