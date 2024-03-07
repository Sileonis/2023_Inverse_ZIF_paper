import os
import pandas as pd
import numpy as np

def splitDataByGas(source_file = './TrainData.xlsx'):
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
        'func3_charge','diffusivity']]

    # Clear NA entries
    cleaned_original_df = cleaned_original_df.dropna()
    # Remove outlier molecule (?)    
    cleaned_original_df=cleaned_original_df.reset_index(drop=True)
   
    uniqueGases = cleaned_original_df.gas.unique()

    print(len(cleaned_original_df.type.unique()))

    print(uniqueGases)

    baseDir = "./DataPerGas"
    if not os.path.exists(baseDir):
        os.mkdir(baseDir)

    for gas in uniqueGases:        
        gasDf = cleaned_original_df[cleaned_original_df['gas'] == gas]
        print("The gas is: " + gas + " And it is connected with " + str(gasDf.shape[0]) + " zifs.")

        gasDf.to_excel(os.path.join(baseDir,str(gas)+".xlsx"), index=False)

if __name__ == "__main__":

    splitDataByGas()