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

####################################################################
# Constant values
#################
linker_length1 = {3.66:{'linker_mass1':83, 'σ':0.325, 'e':0.7112},
                 4.438:{'linker_mass1':81, 'σ':0.25, 'e':0.0627},
                  4.86:{'linker_mass1':101.98, 'σ':0.285, 'e':0.255},
                   5.7:{'linker_mass1':134.906, 'σ':0.34, 'e':1.2552},
                  6.01:{'linker_mass1':223.8, 'σ':0.4, 'e':0.0627},
                  6.41:{'linker_mass1':317.8, 'σ':0.367, 'e':1.8731}
                  }

linker_length2 = {3.66:{'linker_mass2':83, 'σ':0.325, 'e':0.7112},
                 4.438:{'linker_mass2':81, 'σ':0.25, 'e':0.0627},
                  4.86:{'linker_mass2':101.98, 'σ':0.285, 'e':0.255},
                   5.7:{'linker_mass2':134.906, 'σ':0.34, 'e':1.2552},
                  6.01:{'linker_mass2':223.8, 'σ':0.4, 'e':0.0627},
                  6.41:{'linker_mass2':317.8, 'σ':0.367, 'e':1.8731}
                  }
linker_length3 = {3.66:{'linker_mass3':83, 'σ':0.325, 'e':0.7112},
                 4.438:{'linker_mass3':81, 'σ':0.25, 'e':0.0627},
                  4.86:{'linker_mass3':101.98, 'σ':0.285, 'e':0.255},
                   5.7:{'linker_mass3':134.906, 'σ':0.34, 'e':1.2552},
                 5.996:{'linker_mass3':117., 'σ':0.25, 'e':0.0627},
                  6.01:{'linker_mass3':223.8, 'σ':0.4, 'e':0.0627},
                  6.41:{'linker_mass3':317.8, 'σ':0.367, 'e':1.8731}
                  }

func1_length = {2.278:{'func1_mass':1.},
                 3.54:{'func1_mass':35.45},
                  3.78:{'func1_mass':15.},
                   3.85:{'func1_mass':79.9},
                  3.927:{'func1_mass':16.},
                  4.093:{'func1_mass':31.}
                  }

func2_length = {2.278:{'func2_mass':1.},
                 3.54:{'func2_mass':35.45},
                  3.78:{'func2_mass':15.},
                   3.85:{'func2_mass':79.9},
                  3.927:{'func2_mass':16.},
                  4.093:{'func2_mass':31.}
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

####################################################################

def readData(source_file = './MyData.xlsx') -> pd.DataFrame:
    # Read file
    df=pd.read_excel(source_file)
    df.head(2)
    df['logD'] = np.log10(df['diffusivity'])

    # Keep appropriate columns
    df2=df[[ 'type', 'gas', 'MetalNum', 'aperture', 'size - van der Waals (Å)','mass', 'ascentricF', 'logD', 'size - kinetic diameter (Å)', 'ionicRad', 
       'Μ-N_lff', 'Μ-N_kFF', 'MetalCharge', 'MetalMass',
       'apertureAtom_σ', 'apertureAtom_e', 'linker_length1', 'linker_length2',
       'linker_length3', 'linker_mass1', 'linker_mass2', 'linker_mass3',
       'func1_length', 'func2_length', 'func3_length', 'func1_mass',  
       'func2_mass', 'func3_mass', 'func1_charge', 'func2_charge',
       'func3_charge',]]
    
    # Rename columns
    df2=df2.rename(columns={'size - van der Waals (Å)':'diameter', 'size - kinetic diameter (Å)':'kdiameter', 
        'apertureAtom_σ':'σ', 'apertureAtom_e':'e' })

    # Clear NA entries
    df2 = df2.dropna()
    # Remove outlier molecule (?)    
    df2=df2.reset_index(drop=True)
    df2=df2[
    # (df2['gas'] != 'isobutane') & (df2['gas'] != 'iso-butane')
    (df2['gas'] != 'SF6')
    #  (df2['gas'] != 'Rn')
    ].reset_index()

    df2 = df2.dropna()
    df2=df2.reset_index(drop=True)

    

    return df2

def prepareDataForLearning(original_dataframe) -> tuple:
    """Given an appropriate input dataframe of training data, returns a tuple containing: 
     - the dataframe with the appropriate training columns, 
     - the gene representations of the input data, 
     - the representation of the training data inlcuding all dependent and independent features,
     - the target values for each instance of the training data. 
    """
    df3 = original_dataframe[[
     'MetalNum',  
     'linker_length1', 'linker_length2', 'linker_length3',
    'func1_length', 'func2_length', 'func3_length' 
                                 ]]

    df4 = df3.drop_duplicates()
    df4 = df4.reset_index(drop=True)
    
    Genes = np.asanyarray(df4[[
        'MetalNum',  
        'linker_length1', 'linker_length2', 'linker_length3',
        'func1_length', 'func2_length', 'func3_length' 
                                    ]])

    x_all = np.asanyarray(original_dataframe[[ 
        'diameter',
        'mass',
        'ascentricF',
        'kdiameter',
        'MetalNum',  
        'linker_length1', 'linker_length2', 'linker_length3',
        'func1_length', 'func2_length', 'func3_length',
        'ionicRad', 'MetalMass',
        'linker_mass1', 'linker_mass2', 'linker_mass3',
        'σ', 'e', 
        'func1_mass', 'func2_mass', 'func3_mass'
                                ]])


    y_all = np.array(original_dataframe[['logD']])    

    return (df4, Genes,x_all, y_all)

def trainModel(x_all, y_all):
    XGBR = XGBRegressor(n_estimators=500, max_depth=5, eta=0.07, subsample=0.75, colsample_bytree=0.7, reg_lambda=0.4, reg_alpha=0.13,
                        # n_jobs=6,
                        nthread=8,
                        random_state=61
                    )

    model = XGBR.fit(x_all, y_all)

    return model

def get_base_vector_from_solution(solution):
    return np.array([
                  MetalNum[solution[0]]['ionicRad'],
                  MetalNum[solution[0]]['MetalMass'],
                  linker_length1[solution[1]]['linker_mass1'],
                  linker_length2[solution[2]]['linker_mass2'],
                  linker_length3[solution[3]]['linker_mass3'],
                  linker_length1[solution[1]]['σ'],
                  linker_length1[solution[1]]['e'],
                  func1_length[solution[4]]['func1_mass'],
                  func2_length[solution[5]]['func2_mass'],
                  func3_length[solution[6]]['func3_mass']])

def get_whole_vector_per_gas_from_solution(solution, diameter_tuple, mass_tuple, ascF_tuple, kD_tuple):
    diameter_gas1, diameter_gas2 = diameter_tuple
    mass_gas1, mass_gas2 = mass_tuple
    ascF_gas1, ascF_gas2 = ascF_tuple
    kD_gas1, kD_gas2 = kD_tuple

    solution2=get_base_vector_from_solution(solution)
    solution_gas1=np.concatenate((diameter_gas1, mass_gas1, ascF_gas1, kD_gas1, solution,solution2), axis=0, out=None, dtype=None, casting="same_kind")
    solution_gas2=np.concatenate((diameter_gas2, mass_gas2, ascF_gas2, kD_gas2,solution,solution2), axis=0, out=None, dtype=None, casting="same_kind")

    return solution_gas1, solution_gas2


def fitness_base(solution, solution_idx, diameter_tuple, mass_tuple, ascF_tuple, kD_tuple, boundaries_D, boundaries_R, 
        model, customFitnessFormula : Callable[[float, float], float] = None) -> float:
    diameter_gas1, diameter_gas2 = diameter_tuple
    mass_gas1, mass_gas2 = mass_tuple
    ascF_gas1, ascF_gas2 = ascF_tuple
    kD_gas1, kD_gas2 = kD_tuple
        
    # solution2=get_base_vector_from_solution(solution)
    
    solution_gas1, solution_gas2 = get_whole_vector_per_gas_from_solution(solution, diameter_tuple, mass_tuple, ascF_tuple, kD_tuple)
    
    estimated_gas1_diffusivity  = model.predict([solution_gas1])[0]
    estimated_gas2_diffusivity  = model.predict([solution_gas2])[0]

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
        # 'diameter',
        'MetalNum',  
        'linker_length1', 'linker_length2', 'linker_length3',
        'func1_length', 'func2_length', 'func3_length' 
                                    ]
#############

def represent_instances_as_genes(instances_dataframe: pd.DataFrame) -> np.array:
    return np.asanyarray(instances_dataframe[GENE_FIELDS])


def prepareGA(fitness, starting_population_data, **kwargs):
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
        mutation_probability = [0.5, 0.01] # originally [0.4, 0.01]
    else:
        mutation_probability = kwargs['mutation_probability']

    if 'mutation_type' not in kwargs:
        mutation_type = "adaptive" # random, swap, inversion, scramble, adaptive
    else:
        mutation_type = kwargs['mutation_type']

    Genes = np.asanyarray(starting_population_data[GENE_FIELDS])

    initial_population = Genes

    # narrowed down options
    if 'gene_space' not in kwargs:
        gene_space =  [
                    [ 4,12,25,27,29,30,48],
                    [ 4.438, 4.86, 5.7, 6.01, 6.41],
                    [ 4.438, 4.86, 5.7, 6.01, 6.41],
                    [ 4.438, 4.86, 5.7,5.996,6.01, 6.41],
                    [ 3.54, 3.78, 3.85,  4.093],
                    [ 3.54, 3.78, 3.85,  4.093],
                    # [ 3.54, 3.78, 3.85,  4.093]
                    [2.278, 2.7, 3.54, 3.78, 3.85,  4.093, 4.25],
                    ]
    else:
        gene_space = kwargs['gene_space']

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
                       keep_parents=keep_parents,
                       keep_elitism=keep_elitism,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type, 
                       # mutation_num_genes = mutation_num_genes,
                       gene_space = gene_space,
                       # sol_per_pop = sol_per_pop,
                       # num_genes = num_genes,
                       crossover_probability = crossover_probability,
                       mutation_probability = mutation_probability,
                       K_tournament = K_tournament,
                       allow_duplicate_genes=False,
                       # stop_criteria = "reach_100",
                       # stop_criteria=["reach_127.4", "saturate_160"],
                       stop_criteria=stop_criteria,
                       save_solutions=True,
                       random_seed=random_seed,
                    #    parallel_processing=["thread", 20],
                    #    parallel_processing=["process", 8],
                       on_generation=on_generation,
                       save_best_solutions=True
                      )
        
    return ga_instance

def runGA(ga_instance):
    """Returns the solution (as a gene), the related fitness and its index."""

    now = datetime.datetime.now()
    print ("Current date and time : ")
    print (now.strftime("%Y-%m-%d %H:%M:%S"))
    start_time = time.time()
    ga_instance.run()
    print("--- %s minutes ---" % ((time.time() - start_time)/60))
    now = datetime.datetime.now()
    print ("Current date and time : ")
    print (now.strftime("%Y-%m-%d %H:%M:%S"))

    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print("Parameters of the best solution : {solution}".format(solution=solution))
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

    solution2 = get_base_vector_from_solution(solution)

    return solution, solution_fitness, solution_idx

def evaluate_solution(solution, model, diameter_tuple, mass_tuple, ascF_tuple, kD_tuple):
    solution_gas1, solution_gas2= get_whole_vector_per_gas_from_solution(solution, diameter_tuple, mass_tuple, ascF_tuple, kD_tuple)

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

