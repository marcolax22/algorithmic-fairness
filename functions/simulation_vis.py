# -------------------------------------------------------------------------------
# import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------------------------------------------------
# function for plotting the results of the simulation process
def plot_multiple_gender_distribution_kde(*candidate_sets, labels=None, title=None):
    """
    Generate KDE plots for male and female distributions across multiple simulation results.
    
    Parameters:
    -----------
    *candidate_sets : Variable number of DataFrames containing selected candidates
    labels : List of labels for each candidate set
    title : Custom title for the plot
    """
    if not candidate_sets:
        print("No candidate sets provided.")
        return
    
    # Set up default labels if not provided
    if labels is None:
        labels = [f"Simulation {i+1}" for i in range(len(candidate_sets))]
    elif len(labels) < len(candidate_sets):
        # Extend labels if needed
        labels.extend([f"Simulation {i+1}" for i in range(len(labels), len(candidate_sets))])
    
    # Line styles for different simulations
    line_styles = ['-', '--', ':', '-.']
    colors_male = ['blue', 'navy', 'royalblue', 'steelblue']
    colors_female = ['red', 'firebrick', 'salmon', 'tomato']
    
    # Create figure
    plt.figure(figsize=(14, 8))
    
    # Process each candidate set
    for i, candidates in enumerate(candidate_sets):
        if candidates.empty:
            print(f"No candidates in set {i+1}.")
            continue
        
        # Count males and females per iteration
        gender_counts = candidates.groupby(['iteration', 'gender']).size().unstack(fill_value=0)
        
        # Rename columns for clarity if needed
        if 0 in gender_counts.columns and 1 in gender_counts.columns:
            gender_counts = gender_counts.rename(columns={0: 'Male', 1: 'Female'})
        
        # Choose line style and colors (cycle through available styles)
        style_idx = i % len(line_styles)
        color_male_idx = i % len(colors_male)
        color_female_idx = i % len(colors_female)
        
        # Plot male distribution
        if 'Male' in gender_counts.columns:
            male_label = f"{labels[i]} - Male"
            sns.kdeplot(
                gender_counts['Male'], 
                label=male_label, 
                linestyle=line_styles[style_idx],
                color=colors_male[color_male_idx],
                fill=False, 
                alpha=1
            )
            
            # # Add mean line
            # male_mean = gender_counts['Male'].mean()
            # plt.axvline(
            #     male_mean, 
            #     color=colors_male[color_male_idx], 
            #     linestyle=line_styles[style_idx], 
            #     alpha=0.5
            # )
            
            # # Add text annotation
            # y_pos = 0.95 - (i * 0.05)
            # plt.text(
            #     0.05, y_pos, 
            #     f"{male_label} Mean: {male_mean:.2f}", 
            #     transform=plt.gca().transAxes,
            #     color=colors_male[color_male_idx]
            # )
        
        # Plot female distribution
        if 'Female' in gender_counts.columns:
            female_label = f"{labels[i]} - Female"
            sns.kdeplot(
                gender_counts['Female'], 
                label=female_label, 
                linestyle=line_styles[style_idx],
                color=colors_female[color_female_idx],
                fill=False, 
                alpha=1
            )
            
            # # Add mean line
            # female_mean = gender_counts['Female'].mean()
            # plt.axvline(
            #     female_mean, 
            #     color=colors_female[color_female_idx], 
            #     linestyle=line_styles[style_idx], 
            #     alpha=0.5
            # )
            
            # # Add text annotation
            # y_pos = 0.85 - (i * 0.05)
            # plt.text(
            #     0.05, y_pos, 
            #     f"{female_label} Mean: {female_mean:.2f}", 
            #     transform=plt.gca().transAxes,
            #     color=colors_female[color_female_idx]
            # )
    
    # Set title and labels
    if title:
        plt.title(title)
    else:
        plt.title('Distribution of Gender Counts Across Multiple Simulations')
    
    plt.xlabel('Count')
    plt.ylabel('Density')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 50)

    plt.tight_layout()
    plt.show()