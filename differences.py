import os
import numpy as np
import matplotlib.pyplot as plt

def calculate_differences(order: str, interpolation: str, step=20):
    numbers_list = list(range(step, 111, step))

    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # current_dir = 'C:/Users/pkrok/OneDrive/Python/3_Inverse'

    final_file = f'{current_dir}/{order}_{interpolation}_100.npy'
    final_output = np.load(final_file)

    differences = {}
    for number in numbers_list:
        file_path = f'{current_dir}/{order}_{interpolation.capitalize()}_{number}.npy'
        output = np.load(file_path)

        diff = np.abs(output - final_output)
        differences[number] = diff.mean()

    return differences

# Rest of the code remains the same...

# Example usage:
# random_interpolated_diff = calculate_differences('Random', 'interpol', step=20)
random_ml_diff = calculate_differences('Random', 'ML', step=20)
# researcher_interpolated_diff = calculate_differences('Research', 'interpol', step=20)
researcher_ml_diff = calculate_differences('Research', 'ML', step=20)

# Now you have dictionaries containing the differences for each case:
# print("Random Interpolated Differences:")
# print(random_interpolated_diff)

print("Random ML Differences:")
print(random_ml_diff)

# print("Researcher Interpolated Differences:")
# print(researcher_interpolated_diff)

print("Researcher ML Differences:")
print(researcher_ml_diff)

# Plot function
def plot_differences(differences_dict, interpolation: str):
    steps = list(differences_dict[list(differences_dict.keys())[0]].keys())
    random_diff = list(differences_dict['Random'].values())
    researcher_diff = list(differences_dict['Researcher'].values())
    
    plt.figure()
    ax = plt.gca()
    ax.plot(steps, random_diff, label='Random', marker='o', c='r', markersize=14, linewidth=3.5,markeredgewidth=2, markeredgecolor='k')
    ax.plot(steps, researcher_diff, label='Researcher', marker='o', c='g', markersize=14, linewidth=3.5, markeredgewidth=2, markeredgecolor='k')
    
    plt.xlabel('Step (% of dataset)', fontsize=20)
    plt.ylabel('Mean Absolute Difference', fontsize=20)
    # plt.title(f'ML Differences - {interpolation.capitalize()}')
    plt.legend(loc='upper right', fontsize=18, labelspacing=2)

    ax.tick_params(which='both', width=2)
    ax.tick_params(which='major', length=6)
    ax.tick_params(which='minor', length=4, color='r')
    # plt.locator_params(axis='x', numticks =5)
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    
    plt.rcParams["figure.figsize"] = (8, 8)
    
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)

    # Adjust linewidth of the axis spines
    for spine in ax.spines.values():
        spine.set_linewidth(2)  # Adjust the linewidth here as desired

    plt.savefig(f'ML Differences - {interpolation.capitalize()}', bbox_inches="tight", dpi=300)
    plt.show()

# Plot differences for Interpolated
# plot_differences({'Random': random_interpolated_diff, 'Researcher': researcher_interpolated_diff}, 'interpolated')

# Plot differences for ML
plot_differences({'Random': random_ml_diff, 'Researcher': researcher_ml_diff}, 'ML')