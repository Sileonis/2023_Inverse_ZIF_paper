# Source code for the "Inverse Design of ZIFs through Artificial Intelligence Methods" publication
## Overview
The project implements an inverse design process to design functionalized nanoporous materials, of the ZIF family,
that can achieve a user-determined performance (Di, Di/Dj). The code uses a genetic algorithm (GA), based on the PyGAD library, towards
optimizing an ML predictive model, that takes input information on the chemistry/structure of ZIFs and predicts the diffusivity of any given gas.

ML model in use: XGBR regressor

## Setting up
The project runs on Python (>3.8) and requires the following packages: 

    pip install pandas seaborn numpy=1.21 tabulate scipy matplotlib rtree scikit-learn pygad xgboost openpyxl

[//]: # (or)

[//]: # (conda install seaborn numpy=1.21 tabulate scipy matplotlib rtree scikit-learn pygad xgboost openpyxl)

## Running
Once the project packages are in place, you can execute the code to see the program parameters, as follows (from the command line that uses the selected python environment):

    python ga_inverse_testlib_non-jupyter.py -h

## A typical run (for a CO2/CH4 setting) is as follows:

    python ga_inverse_testlib_non-jupyter.py -c co2 -r "2" -lD "-11" -hD "-9" -lR "4" -hR "5" -g "100"

    Among the solutions dFm_Be should appear, which is: 4.0	4.86	3.780

## A typical run (for a O2/N2 setting) is as follows:
    
    python ga_inverse_testlib_non-jupyter.py -c o2 -r "2" -lD "-13" -hD "-11.5" -lR "1" -hR "2" -g "3000"
    Among the solutions Cd-I-ZIF-7-8 should appear, which is: 48.0	4.438	5.996	3.78	4.25

## A typical run (for a C3H6/C3H8 setting) is as follows:
    
    python ga_inverse_testlib_non-jupyter.py -c  -r "2" -lD "-13.5" -hD "-11.5" -lR "1.5" -hR "2" -g "3000"
    Among the solutions ZIF-67 should appear, which is: 27.0	4.438	3.78

## Authors
### Publication
- Panagiotis Krokidas
- Michael Kainourgiakis
- Theodore Steriotis
- George Giannakopoulos
### Source code
- Panagiotis Krokidas
- George Giannakopoulos

# Licence
Apache v2 (TODO: Add file header)