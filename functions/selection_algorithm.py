# -------------------------------------------------------------------------------
# import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# -------------------------------------------------------------------------------
# function for biased selection
def biased_selection(data, male_selection_prob=0.7, female_selection_prob=0.3, 
                     overall_selection_rate=0.25, gender_col='gender', decision_col='decision'):
    
    # Filter candidates where decision column is 1
    candidates = data[data[decision_col] == 1].copy()
    
    if len(candidates) == 0:
        return 0, 0  # No selection possible
    
    # Calculate how many candidates to select
    n_to_select = int(len(candidates) * overall_selection_rate)

    # Assign selection probabilities based on gender
    candidates['selection_prob'] = candidates[gender_col].apply(lambda x: male_selection_prob if x == 1 else female_selection_prob)

    # Perform weighted random selection
    selected_indices = np.random.choice(
        candidates.index, 
        size=n_to_select, 
        replace=False, 
        p=candidates['selection_prob'] / candidates['selection_prob'].sum()
    )

    # Count selected males and females
    selected = candidates.loc[selected_indices]
    male_selected = sum(selected[gender_col] == 0)
    female_selected = sum(selected[gender_col] == 1)
    
    return male_selected, female_selected

# -------------------------------------------------------------------------------
# function for plotting selection results
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def analyze_gender_selection(preds, X_test, selection_function, male_selection_prob=0.7, female_selection_prob=0.3, num_iterations=500):
    """
    Analyzes gender-based selection across multiple prediction columns using Kernel Density Estimation (KDE).
    
    Parameters:
        preds (list): List of prediction column names.
        X_test (DataFrame): The test dataset containing gender information and predictions.
        selection_function (function): Function that applies biased selection.
        male_selection_prob (float): Probability of selecting males.
        female_selection_prob (float): Probability of selecting females.
        num_iterations (int): Number of iterations for collecting selection counts.
    """
    # Initialize dictionaries to store selection counts
    male_counts, female_counts = {key: [] for key in preds}, {key: [] for key in preds}
    
    # Collecting counts for each prediction column
    for _ in range(num_iterations):
        for pred in preds:
            male, female = selection_function(X_test, male_selection_prob=male_selection_prob, female_selection_prob=female_selection_prob, decision_col=pred)
            male_counts[pred].append(male)
            female_counts[pred].append(female)
    
    # Compute Kernel Density Estimation (KDE)
    kde = {
        key: (
            gaussian_kde(male_counts[key], bw_method='silverman'), 
            gaussian_kde(female_counts[key], bw_method='silverman')
        ) for key in preds
    }
    
    # Create x-values for KDE plotting
    x_values = np.linspace(0, max(max(male_counts[key]) for key in preds), 500)
    
    # Define distinct colors for male and female distributions
    male_colors = plt.cm.Blues(np.linspace(0.5, 1, len(preds)))  # Shades of blue for males
    female_colors = plt.cm.Reds(np.linspace(0.5, 1, len(preds)))  # Shades of red for females
    linestyles = ['-', '--', ':', '-.', (0, (3, 5, 1, 5)), (0, (1, 10))]  # Variety of linestyles
    
    # Plot the KDE distributions
    plt.figure(figsize=(12, 6))
    for idx, key in enumerate(preds):
        plt.plot(x_values, kde[key][0](x_values), label=f"Males ({key})", color=male_colors[idx], lw=2, linestyle=linestyles[idx % len(linestyles)])
        plt.plot(x_values, kde[key][1](x_values), label=f"Females ({key})", color=female_colors[idx], lw=2, linestyle=linestyles[idx % len(linestyles)])
    
    plt.xlabel("Number of Selected Candidates")
    plt.ylabel("Density")
    plt.title("Distribution of Selected Candidates (Kernel Density Estimation)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()
    
    # Print the mean and standard deviation for each prediction column
    for key in preds:
        print(f"Mean and Standard Deviation for {key}:")
        print(f"  Male Mean: {np.mean(male_counts[key]):.2f}, Male Std: {np.std(male_counts[key]):.2f}")
        print(f"  Female Mean: {np.mean(female_counts[key]):.2f}, Female Std: {np.std(female_counts[key]):.2f}")

# Example usage:
# preds = ['your_pred_1', 'your_pred_2', 'your_pred_3']
# analyze_gender_selection(preds, X_test, selection.biased_selection, male_selection_prob=0.7, female_selection_prob=0.3)
