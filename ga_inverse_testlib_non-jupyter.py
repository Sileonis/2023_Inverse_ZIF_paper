# Import library
from ga_inverse import *

selected_case = 'co2' # You can choose among 'propylene', 'o2' and 'co2', which correspond to propylene/propane, o2/n2 and co2/ch4 mixtures, respectively
separation = selected_case
diameter_tuple, mass_tuple, ascF_tuple, kD_tuple, linker_length1, func1_length, metalNum, linker_length3, func3_length, GeneFieldNames, gene_space = case(selected_case)

# Read data
data_from_file = readData('./TrainData.xlsx')
training_data, gene_repr_of_training_data, training_x, training_y = prepareDataForLearning(data_from_file, GeneFieldNames)


# Train model
model = train_model(training_x, training_y)

# Prepare GA
## Fitness function
boundaries_D = np.array([-9, -10])
boundaries_R = np.array([4, 5])

# Custom fitness function
def my_fitness(estimated_gas1_diffusivity, estimated_gas2_diffusivity, result_details: dict = None):
    Ratio = estimated_gas1_diffusivity - estimated_gas2_diffusivity
    
    # DiffusivityContribution = 1.0/(abs(np.min(boundaries_D) - estimated_gas1_diffusivity) + abs(np.max(boundaries_D) - estimated_gas1_diffusivity))
    # RatiosContribution = 1.0/(abs(np.min(boundaries_R) - Ratio) + abs(np.max(boundaries_R) - Ratio))

    # If a result_details dictionary exists
    if result_details is not None:
        # then update it the intermediate results
        result_details['Diffusivity'] = estimated_gas1_diffusivity
        result_details['Ratio'] = Ratio
        if  (np.min(boundaries_D) < estimated_gas1_diffusivity < np.max(boundaries_D)) and (np.min(boundaries_R) < Ratio < np.max(boundaries_R)):
            DiffusivityContribution = 1.0/(abs(np.min(boundaries_D) - estimated_gas1_diffusivity) + abs(np.max(boundaries_D) - estimated_gas1_diffusivity))
            RatiosContribution = 1.0/(abs(np.min(boundaries_R) - Ratio) + abs(np.max(boundaries_R) - Ratio))
        else:
            RatiosContribution = 1/(np.exp(abs(Ratio-np.min(boundaries_R)))+np.exp(abs(Ratio-np.max(boundaries_R))))
            DiffusivityContribution = 1/(np.exp(abs(estimated_gas1_diffusivity-np.min(boundaries_D)))+np.exp(abs(estimated_gas1_diffusivity-np.max(boundaries_D))))
    
    overallFitnessMeasure = 0.5*DiffusivityContribution + 0.5*RatiosContribution

    return overallFitnessMeasure

def ga_fitness(solution, solution_idx):
    # Build upon default, imported fitness from library
    return fitness_base(solution=solution, solution_idx=solution_idx, separation=separation,
                        diameter_tuple=diameter_tuple, 
                        mass_tuple=mass_tuple, ascF_tuple=ascF_tuple, kD_tuple=kD_tuple,
                        boundaries_D=boundaries_D, boundaries_R=boundaries_R, 
                        linker_length1=linker_length1, func1_length=func1_length,
                        linker_length3=linker_length3, func3_length=func3_length,
                        model=model, customFitnessFormula=my_fitness)


# Define single loop
def run_one_ga_loop():
    # Starting population
    starting_population = gene_repr_of_training_data

    # Actual initialization
    ga_instance = prepareGA(fitness=ga_fitness, starting_population_data=starting_population,
                            gene_space =  gene_space,
                            num_generations=2, on_generation=None, suppress_warnings=True)
        
    # Run GA
    runGA(ga_instance)

    # Get best solutions
    best_solutions = get_best_solutions(ga_instance)
    print("Best solutions of loop:")
    print(best_solutions)


    # Output also best of the best
    best_solution, best_solution_fitness, _ = ga_instance.best_solution()

    print("Parameters of the best loop solution : %s"%(str(best_solution)))
    print("Fitness value of the best loop solution = %5.3f"%(best_solution_fitness))

    return best_solutions

Rounds = 3
loop_time = 0
total_time = 0

# Init solutions
solutions_from_all_loops = None

for i in range(Rounds):
    start_time = time.time()

    best_solutions_of_loop = run_one_ga_loop()

    # Append to all solutions set
    if solutions_from_all_loops is None:
        solutions_from_all_loops = best_solutions_of_loop
    else:
        solutions_from_all_loops = np.vstack((solutions_from_all_loops, best_solutions_of_loop))
    
    # Update time for loop and total time
    loop_time = (time.time() - start_time)
    total_time = total_time+loop_time
    
    print("Loop time: %fs"%loop_time)
    
    AverLooptime = total_time/Rounds    
    print("Avg. loop time: %fs"%(AverLooptime))

# Save unique best solutions as list
best_solutions_list = np.unique(solutions_from_all_loops,axis=0)


# Output best solutions
for best_solution in best_solutions_list:
    print("Parameters of the best solution : %s"%(str(best_solution)))

# Initialize a dataframe
best_zifs_for_plot = pd.DataFrame(columns = [
        'MOF_ID',
        'logD',
        'Ratio', 'fitness'])

def add_to_best_zifs_list(idx, estimated_gas1_diffusivity, estimated_gas2_diffusivity):
    best_zifs_for_plot.loc[idx, ['MOF_ID']] = idx
    fitness_details = dict()
    best_zifs_for_plot.loc[idx, ['fitness']] = my_fitness(estimated_gas1_diffusivity, estimated_gas2_diffusivity, fitness_details)
    best_zifs_for_plot.loc[idx, ['logD']] = fitness_details['Diffusivity']
    best_zifs_for_plot.loc[idx, ['Ratio']] = fitness_details['Ratio']

    return None


# For each solution
for mof_id, solution_vector in enumerate(best_solutions_list):
    # Calculate the 2 components of the fitness 
    # and add them to the data frame
    fitness_base(solution_vector, mof_id, diameter_tuple, mass_tuple, ascF_tuple, kD_tuple, boundaries_D, boundaries_R, model,
                 customFitnessFormula = lambda estimated_gas1_diffusivity, estimated_gas2_diffusivity: 
                    add_to_best_zifs_list(mof_id, estimated_gas1_diffusivity, estimated_gas2_diffusivity)
                 )
    


# Plot outputs
plot_logDvsRatio(best_zifs_for_plot,'C3H6', 'C3H8')
plot_fitnessPerMOF(best_zifs_for_plot,'C3H6', 'C3H8')
