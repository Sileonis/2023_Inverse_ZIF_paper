# %%
import pandas as pd 
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as  ticker
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from xgboost import XGBRegressor
from mpl_toolkits.axes_grid1 import make_axes_locatable
import random
# from ga_inverse import *

# %% [markdown]
# # Create TypeList

# %%
datas = './TrainData.xlsx'

df=pd.read_excel(datas)
df['logD'] = np.log10(df['diffusivity'])

df2=df[[ 'type', 'gas', 'aperture', 'MetalNum', 'MetalMass', 'size - van der Waals (Å)','mass', 'ascentricF', 'logD', 'size - kinetic diameter (Å)', 'ionicRad', 
       'Μ-N_lff', 'Μ-N_kFF', 'MetalCharge',
       'σ_1', 'e_1', 'linker_length1', 'linker_length2',
       'linker_length3', 'linker_mass1', 'linker_mass2', 'linker_mass3',
       'func1_length', 'func2_length', 'func3_length', 'func1_mass',  
       'func2_mass', 'func3_mass', 'func1_charge', 'func2_charge',
       'func3_charge']]

df2=df2.rename(columns={'size - van der Waals (Å)':'diameter', 'size - kinetic diameter (Å)':'kdiameter', 'apertureAtom_e':'e' })

df2 = df2.dropna()

df2=df2.reset_index(drop=True)

df2.type.unique()

TypeList = df2.type.unique()
TypeList

# %%
def extract_pca_data_space(MOF_vectors, gas_name, trained_model_for_specific_gas):

    # solution = np.asanyarray(MOF_vectors[['MetalNum', 'linker_length1', 'func1_length']])

    if gas_name == 'propylene':
        diameter = np.array([4.03])
        mass = np.array([42.08])
        ascentricF = np.array([0.142])
        kdiameter = np.array([4.5])
    elif gas_name == 'co2':
        diameter = np.array([3.24])
        mass = np.array([44.01])
        ascentricF = np.array([0.225])
        kdiameter = np.array([3.3])
    
    # Provided dictionaries
    linker_length1 = {
        3.66: {'linker_mass1': 83, 'σ_1': 0.325, 'e_1': 0.7112, 'linker_length2': 3.66, 'linker_mass2': 83, 'σ_2': 0.325,
            'e_2': 0.7112, 'linker_length3': 3.66, 'linker_mass3': 83, 'σ_3': 0.325, 'e_3': 0.7112},
        4.438: {'linker_mass1': 81, 'σ_1': 0.25, 'e_1': 0.0627, 'linker_length2': 4.438, 'linker_mass2': 81, 'σ_2': 0.25,
                'e_2': 0.0627, 'linker_length3': 4.438, 'linker_mass3': 81, 'σ_3': 0.25, 'e_3': 0.0627},
        4.86: {'linker_mass1': 101.98, 'σ_1': 0.285, 'e_1': 0.255, 'linker_length2': 4.86, 'linker_mass2': 101.98, 'σ_2': 0.285,
            'e_2': 0.255, 'linker_length3': 4.86, 'linker_mass3': 101.98, 'σ_3': 0.285, 'e_3': 0.255},
        5.7: {'linker_mass1': 134.906, 'σ_1': 0.34, 'e_1': 1.2552, 'linker_length2': 5.7, 'linker_mass2': 134.906, 'σ_2': 0.34,
            'e_2': 1.2552, 'linker_length3': 5.7, 'linker_mass3': 134.906, 'σ_3': 0.34, 'e_3': 1.2552},
        6.01: {'linker_mass1': 223.8, 'σ_1': 0.4, 'e_1': 1.8731, 'linker_length2': 6.01, 'linker_mass2': 223.8, 'σ_2': 0.4,
            'e_2': 1.8731, 'linker_length3': 6.01, 'linker_mass3': 223.8, 'σ_3': 0.4, 'e_3': 1.8731},
        6.41: {'linker_mass1': 317.8, 'σ_1': 0.367, 'e_1': 2.4267, 'linker_length2': 6.41, 'linker_mass2': 317.8, 'σ_2': 0.367,
            'e_2': 2.4267, 'linker_length3': 6.01, 'linker_mass3': 223.8, 'σ_3': 0.367, 'e_3': 2.4267}
    }

    func1_length = {
        2.278: {'func1_mass': 1., 'func2_length': 2.278, 'func2_mass': 1., 'func3_length': 2.278, 'func3_mass': 1.},
        3.54: {'func1_mass': 35.45, 'func2_length': 3.54, 'func2_mass': 35.45, 'func3_length': 3.54, 'func3_mass': 35.45},
        3.78: {'func1_mass': 15., 'func2_length': 3.78, 'func2_mass': 15., 'func3_length': 3.78, 'func3_mass': 15.},
        3.85: {'func1_mass': 79.9, 'func2_length': 3.85, 'func2_mass': 79.9, 'func3_length': 3.85, 'func3_mass': 79.9},
        3.927: {'func1_mass': 16., 'func2_length': 3.927, 'func2_mass': 16., 'func3_length': 3.927, 'func3_mass': 16.},
        4.093: {'func1_mass': 31., 'func2_length': 4.093, 'func2_mass': 31., 'func3_length': 4.093, 'func3_mass': 31.}
    }

    MetalNum = {
        4: {'ionicRad': 41, 'MetalMass': 9.012},
        29: {'ionicRad': 71, 'MetalMass': 63.456},
        12: {'ionicRad': 71, 'MetalMass': 24.305},
        27: {'ionicRad': 72, 'MetalMass': 58.930},
        30: {'ionicRad': 74, 'MetalMass': 65.380},
        25: {'ionicRad': 80, 'MetalMass': 54.938},
        48: {'ionicRad': 92, 'MetalMass': 112.411}
    }

    # Function to map values from dictionaries to create additional columns
    def map_values(row):
        metal_num_info = MetalNum[row['MetalNum']]
        linker_info = linker_length1[row['linker_length1']]
        func_info = func1_length[row['func1_length']]
        
        return pd.Series({
            'linker_mass1': linker_info['linker_mass1'],
            'σ_1': linker_info['σ_1'],
            'e_1': linker_info['e_1'],
            'linker_length2': linker_info['linker_length2'],
            'linker_mass2': linker_info['linker_mass2'],
            'σ_2': linker_info['σ_2'],
            'e_2': linker_info['e_2'],
            'linker_length3': linker_info['linker_length3'],
            'linker_mass3': linker_info['linker_mass3'],
            'σ_3': linker_info['σ_3'],
            'e_3': linker_info['e_3'],
            'func1_mass': func_info['func1_mass'],
            'func2_length': func_info['func2_length'],
            'func2_mass': func_info['func2_mass'],
            'func3_length': func_info['func3_length'],
            'func3_mass': func_info['func3_mass'],
            'ionicRad': metal_num_info['ionicRad'],
            'MetalMass': metal_num_info['MetalMass']
        })

    # Apply the mapping function to create new columns in the DataFrame
    new_columns = MOF_vectors.apply(map_values, axis=1)

    # Concatenate the new columns with the original DataFrame
    new_df = pd.concat([MOF_vectors, new_columns], axis=1)

    # Reset the index to avoid duplicate rows
    new_df = new_df.reset_index(drop=True)

    new_df = new_df[['ionicRad', 
                    'MetalNum',
                    'MetalMass',
        'σ_1', 'e_1',
        'σ_2', 'e_2',
        'σ_3', 'e_3',
        'linker_length1', 'linker_length2', 'linker_length3',
        'linker_mass1', 'linker_mass2', 'linker_mass3',
        'func1_length', 'func2_length', 'func3_length', 
        'func1_mass', 'func2_mass', 'func3_mass']]

    # Add new columns to the DataFrame with constant values
    new_df['diameter'] = diameter[0]
    new_df['mass'] = mass[0]
    new_df['ascentricF'] = ascentricF[0]
    new_df['kdiameter'] = kdiameter[0]

    # print("Number of NaN or infinite values in new_df:", np.sum(np.isnan(new_df)) + np.sum(np.isinf(new_df)))
    print("Number of unique rows in new_df:", len(np.unique(new_df, axis=0)))
    new_df=new_df.drop_duplicates()
    MOF_and_gas_vector=new_df.values
    MOFs=MOF_vectors[['MetalNum', 'linker_length1', 'func1_length']].drop_duplicates()

    # After creating MOF_and_gas_vector, add the following code to check for NaN or infinite values
    print("Number of NaN or infinite values in MOF_and_gas_vector:", np.sum(np.isnan(MOF_and_gas_vector)) + np.sum(np.isinf(MOF_and_gas_vector)))

    # Before applying PCA, add the following code to scale the data and check for NaN or infinite values again
    X_scaled = StandardScaler().fit_transform(MOFs)
    print("Number of NaN or infinite values in X_scaled:", np.sum(np.isnan(X_scaled)) + np.sum(np.isinf(X_scaled)))
    duplicates_mask = new_df.duplicated()
    print("Number of duplicates in MOF_and_gas_vector:", np.sum(duplicates_mask))   
    # Apply PCA to the input data
    X_scaled = StandardScaler().fit_transform(MOFs)
    pca = PCA(n_components=2).fit(X_scaled)
    
   
    # For each possible MOF
    # Call the model
    y_pred = trained_model_for_specific_gas.predict(MOF_and_gas_vector)
    print("Number of unique rows in y_pred:", len(np.unique(y_pred, axis=0)))

    # Map X to the new space
    MOF_and_gas_vector_in_PCA_space = pca.transform(X_scaled)
    # After applying PCA, add the following code to check for any duplicate rows in the PCA-transformed data
    print("Number of unique rows in MOF_and_gas_vector_in_PCA_space:", len(np.unique(MOF_and_gas_vector_in_PCA_space, axis=0)))
    print("Number of NaN or infinite values in MOF_and_gas_vector_in_PCA_space:", np.sum(np.isnan(MOF_and_gas_vector_in_PCA_space)) + np.sum(np.isinf(MOF_and_gas_vector_in_PCA_space)))
    print("First few rows of MOF_and_gas_vector_in_PCA_space:")
    print(MOF_and_gas_vector_in_PCA_space[:5])
    # Create DataFrame for PCA components
    final_data_pca = pd.DataFrame(columns=['PC1', 'PC2'])
    final_data_pca['PC1'] = MOF_and_gas_vector_in_PCA_space[:, 0]
    final_data_pca['PC2'] = MOF_and_gas_vector_in_PCA_space[:, 1]

    # Create DataFrame for predicted values
    final_data_output = pd.DataFrame(columns=['output'])
    final_data_output['output'] = y_pred
    print("First few model predictions (y_pred):")
    print(y_pred[:5])
    # Concatenate the DataFrames along columns (axis=1)
    final_data = pd.concat([final_data_pca, final_data_output], axis=1)

    return final_data
    
def get_trained_model(training_data):
    datas = training_data # './TrainData.xlsx'
    df=pd.read_excel(datas)
    df['logD'] = np.log10(df['diffusivity'])

    df2=df[[ 'type', 'gas', 'aperture', 'MetalNum', 'MetalMass', 'size - van der Waals (Å)','mass', 'ascentricF', 'logD', 'size - kinetic diameter (Å)', 'ionicRad', 
       'Μ-N_lff', 'Μ-N_kFF', 'MetalCharge',
       'σ_1', 'e_1', 'σ_2', 'e_2', 'σ_3', 'e_3', 'linker_length1', 'linker_length2',
       'linker_length3', 'linker_mass1', 'linker_mass2', 'linker_mass3',
       'func1_length', 'func2_length', 'func3_length', 'func1_mass',  
       'func2_mass', 'func3_mass', 'func1_charge', 'func2_charge',
       'func3_charge']]
    
    df2=df2.rename(columns={'size - van der Waals (Å)':'diameter', 'size - kinetic diameter (Å)':'kdiameter' })
    df2 = df2.dropna()
    df2=df2.reset_index(drop=True)

    x = np.asanyarray(df2[[
    'ionicRad',
    'MetalNum',
    'MetalMass',
    'σ_1', 'e_1',
    'σ_2', 'e_2',
    'σ_3', 'e_3',
    'linker_length1', 'linker_length2', 'linker_length3',
    'linker_mass1', 'linker_mass2', 'linker_mass3',
    'func1_length', 'func2_length', 'func3_length', 
    'func1_mass', 'func2_mass', 'func3_mass',
    'diameter',
    'mass',
    'ascentricF',
    'kdiameter'
                      ]])

    y = np.array(df2[['logD']])

    XGBR = XGBRegressor(n_estimators=500, max_depth=5, eta=0.07, subsample=0.75, colsample_bytree=0.7, reg_lambda=0.4, reg_alpha=0.13,
                    # n_jobs=6,
                    nthread=6,
                    random_state=6410
                   )
    
    model = XGBR.fit(x, y)
    
    return model

def get_exhaustive_search_data():
    datas = './results.csv'
    df=pd.read_csv(datas, header=None)
    df = df.rename(columns={0: 'MetalNum', 1: 'linker_length1', 2: 'func1_length', 3:'logD1', 4:'logD2', 5: 'Ratio'}) # TODO: Fix
    print("Number of unique rows in df:", len(np.unique(df, axis=0)))
    return df

# %%
# Tests
# extract_pca_data_space(MOF_vectors, gas_name, trained_model_for_specific_gas)
# extract_pca_data_space(get_exhaustive_search_data(), "co2", get_trained_model("./TrainData.xlsx"))

# %% [markdown]
# # Interpolation

# %% [markdown]
# # Define my functions to loop through a varying/increasing training dataset

# %%
def extract_pca_data_space(MOF_vectors, gas_name, trained_model_for_specific_gas):

    # solution = np.asanyarray(MOF_vectors[['MetalNum', 'linker_length1', 'func1_length']])

    if gas_name == 'propylene':
        diameter = np.array([4.03])
        mass = np.array([42.08])
        ascentricF = np.array([0.142])
        kdiameter = np.array([4.5])
    elif gas_name == 'co2':
        diameter = np.array([3.24])
        mass = np.array([44.01])
        ascentricF = np.array([0.225])
        kdiameter = np.array([3.3])
    
    # Provided dictionaries
    linker_length1 = {
        3.66: {'linker_mass1': 83, 'σ_1': 0.325, 'e_1': 0.7112, 'linker_length2': 3.66, 'linker_mass2': 83, 'σ_2': 0.325,
            'e_2': 0.7112, 'linker_length3': 3.66, 'linker_mass3': 83, 'σ_3': 0.325, 'e_3': 0.7112},
        4.438: {'linker_mass1': 81, 'σ_1': 0.25, 'e_1': 0.0627, 'linker_length2': 4.438, 'linker_mass2': 81, 'σ_2': 0.25,
                'e_2': 0.0627, 'linker_length3': 4.438, 'linker_mass3': 81, 'σ_3': 0.25, 'e_3': 0.0627},
        4.86: {'linker_mass1': 101.98, 'σ_1': 0.285, 'e_1': 0.255, 'linker_length2': 4.86, 'linker_mass2': 101.98, 'σ_2': 0.285,
            'e_2': 0.255, 'linker_length3': 4.86, 'linker_mass3': 101.98, 'σ_3': 0.285, 'e_3': 0.255},
        5.7: {'linker_mass1': 134.906, 'σ_1': 0.34, 'e_1': 1.2552, 'linker_length2': 5.7, 'linker_mass2': 134.906, 'σ_2': 0.34,
            'e_2': 1.2552, 'linker_length3': 5.7, 'linker_mass3': 134.906, 'σ_3': 0.34, 'e_3': 1.2552},
        6.01: {'linker_mass1': 223.8, 'σ_1': 0.4, 'e_1': 1.8731, 'linker_length2': 6.01, 'linker_mass2': 223.8, 'σ_2': 0.4,
            'e_2': 1.8731, 'linker_length3': 6.01, 'linker_mass3': 223.8, 'σ_3': 0.4, 'e_3': 1.8731},
        6.41: {'linker_mass1': 317.8, 'σ_1': 0.367, 'e_1': 2.4267, 'linker_length2': 6.41, 'linker_mass2': 317.8, 'σ_2': 0.367,
            'e_2': 2.4267, 'linker_length3': 6.01, 'linker_mass3': 223.8, 'σ_3': 0.367, 'e_3': 2.4267}
    }

    func1_length = {
        2.278: {'func1_mass': 1., 'func2_length': 2.278, 'func2_mass': 1., 'func3_length': 2.278, 'func3_mass': 1.},
        3.54: {'func1_mass': 35.45, 'func2_length': 3.54, 'func2_mass': 35.45, 'func3_length': 3.54, 'func3_mass': 35.45},
        3.78: {'func1_mass': 15., 'func2_length': 3.78, 'func2_mass': 15., 'func3_length': 3.78, 'func3_mass': 15.},
        3.85: {'func1_mass': 79.9, 'func2_length': 3.85, 'func2_mass': 79.9, 'func3_length': 3.85, 'func3_mass': 79.9},
        3.927: {'func1_mass': 16., 'func2_length': 3.927, 'func2_mass': 16., 'func3_length': 3.927, 'func3_mass': 16.},
        4.093: {'func1_mass': 31., 'func2_length': 4.093, 'func2_mass': 31., 'func3_length': 4.093, 'func3_mass': 31.}
    }

    MetalNum = {
        4: {'ionicRad': 41, 'MetalMass': 9.012},
        29: {'ionicRad': 71, 'MetalMass': 63.456},
        12: {'ionicRad': 71, 'MetalMass': 24.305},
        27: {'ionicRad': 72, 'MetalMass': 58.930},
        30: {'ionicRad': 74, 'MetalMass': 65.380},
        25: {'ionicRad': 80, 'MetalMass': 54.938},
        48: {'ionicRad': 92, 'MetalMass': 112.411}
    }

    # Function to map values from dictionaries to create additional columns
    def map_values(row):
        metal_num_info = MetalNum[row['MetalNum']]
        linker_info = linker_length1[row['linker_length1']]
        func_info = func1_length[row['func1_length']]
        
        return pd.Series({
            'linker_mass1': linker_info['linker_mass1'],
            'σ_1': linker_info['σ_1'],
            'e_1': linker_info['e_1'],
            'linker_length2': linker_info['linker_length2'],
            'linker_mass2': linker_info['linker_mass2'],
            'σ_2': linker_info['σ_2'],
            'e_2': linker_info['e_2'],
            'linker_length3': linker_info['linker_length3'],
            'linker_mass3': linker_info['linker_mass3'],
            'σ_3': linker_info['σ_3'],
            'e_3': linker_info['e_3'],
            'func1_mass': func_info['func1_mass'],
            'func2_length': func_info['func2_length'],
            'func2_mass': func_info['func2_mass'],
            'func3_length': func_info['func3_length'],
            'func3_mass': func_info['func3_mass'],
            'ionicRad': metal_num_info['ionicRad'],
            'MetalMass': metal_num_info['MetalMass']
        })

    # Apply the mapping function to create new columns in the DataFrame
    new_columns = MOF_vectors.apply(map_values, axis=1)

    # Concatenate the new columns with the original DataFrame
    new_df = pd.concat([MOF_vectors, new_columns], axis=1)

    # Reset the index to avoid duplicate rows
    new_df = new_df.reset_index(drop=True)

    new_df = new_df[['ionicRad', 
                    'MetalNum',
                    'MetalMass',
        'σ_1', 'e_1',
        'σ_2', 'e_2',
        'σ_3', 'e_3',
        'linker_length1', 'linker_length2', 'linker_length3',
        'linker_mass1', 'linker_mass2', 'linker_mass3',
        'func1_length', 'func2_length', 'func3_length', 
        'func1_mass', 'func2_mass', 'func3_mass']]

    # Add new columns to the DataFrame with constant values
    new_df['diameter'] = diameter[0]
    new_df['mass'] = mass[0]
    new_df['ascentricF'] = ascentricF[0]
    new_df['kdiameter'] = kdiameter[0]

    # print("Number of NaN or infinite values in new_df:", np.sum(np.isnan(new_df)) + np.sum(np.isinf(new_df)))
    print("Number of unique rows in new_df:", len(np.unique(new_df, axis=0)))
    new_df=new_df.drop_duplicates()
    MOF_and_gas_vector=new_df.values
    MOFs=MOF_vectors[['MetalNum', 'linker_length1', 'func1_length']].drop_duplicates()

    # After creating MOF_and_gas_vector, add the following code to check for NaN or infinite values
    print("Number of NaN or infinite values in MOF_and_gas_vector:", np.sum(np.isnan(MOF_and_gas_vector)) + np.sum(np.isinf(MOF_and_gas_vector)))

    # Before applying PCA, add the following code to scale the data and check for NaN or infinite values again
    X_scaled = StandardScaler().fit_transform(MOFs)
    print("Number of NaN or infinite values in X_scaled:", np.sum(np.isnan(X_scaled)) + np.sum(np.isinf(X_scaled)))
    duplicates_mask = new_df.duplicated()
    print("Number of duplicates in MOF_and_gas_vector:", np.sum(duplicates_mask))   
    # Apply PCA to the input data
    X_scaled = StandardScaler().fit_transform(MOFs)
    pca = PCA(n_components=2).fit(X_scaled)
    
   
    # For each possible MOF
    # Call the model
    y_pred = trained_model_for_specific_gas.predict(MOF_and_gas_vector)
    print("Number of unique rows in y_pred:", len(np.unique(y_pred, axis=0)))

    # Map X to the new space
    MOF_and_gas_vector_in_PCA_space = pca.transform(X_scaled)
    # After applying PCA, add the following code to check for any duplicate rows in the PCA-transformed data
    print("Number of unique rows in MOF_and_gas_vector_in_PCA_space:", len(np.unique(MOF_and_gas_vector_in_PCA_space, axis=0)))
    print("Number of NaN or infinite values in MOF_and_gas_vector_in_PCA_space:", np.sum(np.isnan(MOF_and_gas_vector_in_PCA_space)) + np.sum(np.isinf(MOF_and_gas_vector_in_PCA_space)))
    print("First few rows of MOF_and_gas_vector_in_PCA_space:")
    print(MOF_and_gas_vector_in_PCA_space[:5])
    # Create DataFrame for PCA components
    final_data_pca = pd.DataFrame(columns=['PC1', 'PC2'])
    final_data_pca['PC1'] = MOF_and_gas_vector_in_PCA_space[:, 0]
    final_data_pca['PC2'] = MOF_and_gas_vector_in_PCA_space[:, 1]

    # Create DataFrame for predicted values
    final_data_output = pd.DataFrame(columns=['output'])
    final_data_output['output'] = y_pred
    print("First few model predictions (y_pred):")
    print(y_pred[:5])
    # Concatenate the DataFrames along columns (axis=1)
    final_data = pd.concat([final_data_pca, final_data_output], axis=1)

    return final_data
    
def get_trained_model(training_data, randomized_order: bool, perc=10):
    datas = training_data # './TrainData.xlsx'
    df=pd.read_excel(datas)
    df['logD'] = np.log10(df['diffusivity'])

    df2=df[[ 'type', 'gas', 'aperture', 'MetalNum', 'MetalMass', 'size - van der Waals (Å)','mass', 'ascentricF', 'logD', 'size - kinetic diameter (Å)', 'ionicRad', 
       'Μ-N_lff', 'Μ-N_kFF', 'MetalCharge',
       'σ_1', 'e_1', 'σ_2', 'e_2', 'σ_3', 'e_3', 'linker_length1', 'linker_length2',
       'linker_length3', 'linker_mass1', 'linker_mass2', 'linker_mass3',
       'func1_length', 'func2_length', 'func3_length', 'func1_mass',  
       'func2_mass', 'func3_mass', 'func1_charge', 'func2_charge',
       'func3_charge']]
    
    df2=df2.rename(columns={'size - van der Waals (Å)':'diameter', 'size - kinetic diameter (Å)':'kdiameter' })
    df2 = df2.dropna()
    df2=df2.reset_index(drop=True)

    newDf = None

    # Create a list containing a randomized or serial version of the training instance indices
    if randomized_order:
        percentage = perc  # Adjust the percentage
        random_sample = np.random.choice(TypeList, size=int(len(TypeList) * percentage / 100), replace=False)

        # Create a new DataFrame containing only the rows with 'type' values in the TypeList array
        newDf = df2[df2['type'].isin(random_sample)]
    else:
        percentage = perc  # Adjust the percentage
        start_idx = 0
        end_idx = int(len(TypeList) * percentage / 100)
        sequential_sample = TypeList[start_idx:end_idx]

        # Create a new DataFrame containing only the rows with 'type' values in the sequential_sample array
        newDf = df2[df2['type'].isin(sequential_sample)]


    x = np.asanyarray(newDf[[
    'ionicRad',
    'MetalNum',
    'MetalMass',
    'σ_1', 'e_1',
    'σ_2', 'e_2',
    'σ_3', 'e_3',
    'linker_length1', 'linker_length2', 'linker_length3',
    'linker_mass1', 'linker_mass2', 'linker_mass3',
    'func1_length', 'func2_length', 'func3_length', 
    'func1_mass', 'func2_mass', 'func3_mass',
    'diameter',
    'mass',
    'ascentricF',
    'kdiameter'
                    ]])

    y = np.array(newDf[['logD']])

    XGBR = XGBRegressor(n_estimators=500, max_depth=5, eta=0.07, subsample=0.75, colsample_bytree=0.7, reg_lambda=0.4, reg_alpha=0.13,
                    # n_jobs=6,
                    nthread=6,
                    random_state=6410
                )
    
    model = XGBR.fit(x, y)
        
    return model

def get_exhaustive_search_data():
    datas = './results.csv'
    df=pd.read_csv(datas, header=None)
    df = df.rename(columns={0: 'MetalNum', 1: 'linker_length1', 2: 'func1_length', 3:'logD1', 4:'logD2', 5: 'Ratio'}) # TODO: Fix
    print("Number of unique rows in df:", len(np.unique(df, axis=0)))
    return df

# %%
# Test the extract function
# extract_pca_data_space(get_exhaustive_search_data(), "co2", get_trained_model("./TrainData.xlsx", False, 10))

# %% [markdown]
# # Define my PCA Plotting Function

# %%
import scipy.interpolate

def plot_pca_data(pca_df, title, randomized_order: bool, number):
    # Extract the PCA components as a 2D array
    pca_array = pca_df[['PC1', 'PC2']].values

    # Extract the output property as a 1D array
    output_property = pca_df['output'].values

    # Create a figure and axes
    fig, ax = plt.subplots()

    # Plot the scatter plot of PCA axes with larger point size and thin edge lines
    scatter = ax.scatter(pca_array[:, 0], pca_array[:, 1], c=output_property, cmap='coolwarm', s=100, linewidths=0.2, edgecolors='black')

    # Set the labels and title
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title(title)

    # Find minima, maxima
    minXVal = pca_array[:, 0].min() - 0.2
    maxXVal = pca_array[:, 0].max() + 0.2
    minYVal = pca_array[:, 1].min() - 0.2
    maxYVal = pca_array[:, 1].max() + 0.2

    # Increase the scale of both axes
    plt.xlim(minXVal, maxXVal)
    plt.ylim(minYVal, maxYVal)

    # Create a colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.1)
    
    # Get the minimum and maximum values of the output_property for the interpolation range
    min_output_property = output_property.min()
    max_output_property = output_property.max() 

    # Perform the interpolation using the same range as the original points
    # sorted_x_indices = pca_array[:, 0].argsort()
    # sorted_x = pca_array[:, 0][sorted_x_indices]
    # sorted_y = pca_array[:, 1][sorted_x_indices]
    # values = output_property[sorted_x_indices]
    X = pca_array[:, 0]
    Y = pca_array[:, 1]
    Z = output_property

    # interp_func = scipy.interpolate.bisplrep(sorted_x, sorted_y, sorted_output_property)
    # interp_output_property = scipy.interpolate.bisplev(sorted_x, sorted_y, interp_func)
    interp_X = np.linspace(min(X), max(X), num=500)
    interp_Y = np.linspace(min(Y), max(Y), num=500)
    gridX, gridY = np.meshgrid(interp_X, interp_Y)  # 2D grid for interpolation
    interp = scipy.interpolate.LinearNDInterpolator(list(zip(X, Y)), Z)
    interpZ = interp(gridX, gridY)

    # Plot the interpolated heatmap
    # heatmap = plt.pcolormesh(gridX, gridY, interpZ, shading='auto')
    # plt.colorbar()
    heatmap = ax.imshow(interpZ, cmap='coolwarm', extent=[interp_X.min(), interp_X.max(), interp_Y.min(), interp_Y.max()], origin='lower')
    cbar = plt.colorbar(heatmap, cax=cax, label='log$D$')
    cbar.ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))  # Set the colorbar ticks to rounded integers   

    # Save the plot with the specified filename, based on the percentage
    if randomized_order:
        save_filename = f'PCA_Random_{number}.png'
    else:
        save_filename = f'PCA_Researchers_{number}.png'
    plt.savefig(save_filename, bbox_inches='tight', dpi=300)

    # Show the plot (optional)
    plt.show()

# %% [markdown]
# # Define the Final Function: run'n'plot

# %%
def run_n_plot(order: bool, step= 20):
    numbers_list = list(range(step, 111, step))
    OrderRandom = order

    # Loop through each number in the list, which corresponds to a percentage of the training set
    for number in numbers_list:
        # Get the PCA data for the current number
        pca_df = extract_pca_data_space(get_exhaustive_search_data(), "co2", get_trained_model("./TrainData.xlsx", OrderRandom, number))
        
        # Generate and save the PCA plot for the current number
        # Generate and save the PCA plot for the current number
        if OrderRandom:
            plot_pca_data(pca_df, f'Random - Trained on %{number} of original set', OrderRandom, number)
        else:
            plot_pca_data(pca_df, f'Researcher - Trained on %{number} of original set', OrderRandom, number)

# %%
run_n_plot(False, 20)

# %% [markdown]
# # Ploting all in a common figure

# %%
def draw_common_figure():
    # TODO: Implement (take the images and stitch them together?)
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from scipy.interpolate import griddata

    numbers_list = list(range(0, 111, 20))
    OrderRandom = False

    # Create a common figure with subplots
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))

    # Loop through each number in the list, which corresponds to a percentage of the training set
    for i, number in enumerate(numbers_list):
        # Get the PCA data for the current number
        pca_df = extract_pca_data_space(get_exhaustive_search_data(), "co2", get_trained_model("./TrainData.xlsx", OrderRandom, number))

        # Generate the PCA plot for the current number
        pca_array = pca_df[['PC1', 'PC2']].values
        output_property = pca_df['output'].values

        # Generate a grid for interpolation
        x_grid, y_grid = np.meshgrid(np.linspace(pca_array[:, 0].min(), pca_array[:, 0].max(), 100),
                                    np.linspace(pca_array[:, 1].min(), pca_array[:, 1].max(), 100))
        points = pca_array
        values = output_property
        interpolated_values = griddata(points, values, (x_grid, y_grid), method='linear')

        # Plot the scatter plot of PCA axes with larger point size
        scatter = axs[i // 3, i % 3].scatter(pca_array[:, 0], pca_array[:, 1], c=output_property, cmap='coolwarm', s=100, linewidths=0.2, edgecolors='black')
        axs[i // 3, i % 3].imshow(interpolated_values.T, extent=(pca_array[:, 0].min(), pca_array[:, 0].max(),
                                                                pca_array[:, 1].min(), pca_array[:, 1].max()),
                                aspect='auto', origin='lower', alpha=0.5, cmap='coolwarm')

        # Set the labels and title
        axs[i // 3, i % 3].set_xlabel('Principal Component 1')
        axs[i // 3, i % 3].set_ylabel('Principal Component 2')
        axs[i // 3, i % 3].set_title(f'PCA Plot with Heatmap (Number: {number})')

        # Create a colorbar
        divider = make_axes_locatable(axs[i // 3, i % 3])
        cax = divider.append_axes('right', size='5%', pad=0.1)
        cbar = plt.colorbar(scatter, cax=cax, label='log$D$', ax=axs[i // 3, i % 3])

    # Adjust spacing between subplots
    plt.tight_layout()

    # Save the common figure
    plt.savefig('PCA_Plots_Common.png', bbox_inches='tight', dpi=300)

    # Show the common figure
    plt.show()

