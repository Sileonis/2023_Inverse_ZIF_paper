# Main imports
import pandas as pd
import numpy as np

# Learning
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar
#import models
# from sklearn.linear_model import LinearRegression
# from sklearn.neural_network import MLPRegressor
# from sklearn.tree import DecisionTreeRegressor
# from sklearn import linear_model
# from sklearn.ensemble import RandomForestRegressor
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
from rtree import index

# Search / optimization
import pygad

# Utility
import time
import datetime


def readData(source_file = './MyData.xlsx') -> pd.DataFrame:
    # Read file
    df=pd.read_excel(source_file)
    df.head(2)
    df['logD'] = np.log10(df['diffusivity'])
    # df = df.dropna()
    # df[df.gas == 'propylene']['logD'].max()
    df[df.gas == 'N2']['logD'].count()

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
    # 'diameter',
     'MetalNum',  
     'linker_length1', 'linker_length2', 'linker_length3',
    'func1_length', 'func2_length', 'func3_length' 
                                 ]]

    df4 = df3.drop_duplicates()
    df4 = df4.reset_index(drop=True)
    
    Genes = np.asanyarray(df4[[
        # 'diameter',
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


def fitness(solution, solution_idx, diameter_tuple, mass_tuple, ascF_tuple, kD_tuple, boundaries_D, boundaries_R, 
        model, customFitnessFormula : function[float, float] = None) -> float:
    diameter_gas1, diameter_gas2 = diameter_tuple
    mass_gas1, mass_gas2 = mass_tuple
    ascF_gas1, ascF_gas2 = ascF_tuple
    kD_gas1, kD_gas2 = kD_tuple
        
    solution2=np.array([
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
    
    solution_gas1=np.concatenate((diameter_gas1, mass_gas1, ascF_gas1, kD_gas1, solution,solution2), axis=0, out=None, dtype=None, casting="same_kind")
    solution_gas2=np.concatenate((diameter_gas2, mass_gas2, ascF_gas2, kD_gas2,solution,solution2), axis=0, out=None, dtype=None, casting="same_kind")
    
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