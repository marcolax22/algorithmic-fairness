# -------------------------------------------------------------------------------
# import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------------------------------------------------
# function for plotting the results of the simulation process
def plot_multiple_gender_distribution_kde_with_ax(*candidate_sets, labels=None, title=None, ax=None, show_legend=True):
    """Modified version that accepts an axis parameter and allows toggling the legend."""
    if not candidate_sets:
        print("No candidate sets provided.")
        return
    
    # Set up default labels if not provided
    if labels is None:
        labels = [f"Simulation {i+1}" for i in range(len(candidate_sets))]
    elif len(labels) < len(candidate_sets):
        labels.extend([f"Simulation {i+1}" for i in range(len(labels), len(candidate_sets))])
    
    # Line styles for different simulations
    line_styles = ['-', '--', ':', '-.']
    colors_male = ['blue', 'navy', 'royalblue', 'steelblue']
    colors_female = ['red', 'firebrick', 'salmon', 'tomato']
    
    # Use provided axis or create new figure
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot each candidate set
    for i, candidates in enumerate(candidate_sets):
        if candidates.empty:
            print(f"No candidates in set {i+1}.")
            continue
        
        gender_counts = candidates.groupby(['iteration', 'gender']).size().unstack(fill_value=0)
        if 0 in gender_counts.columns and 1 in gender_counts.columns:
            gender_counts = gender_counts.rename(columns={0: 'Male', 1: 'Female'})
        
        style_idx = i % len(line_styles)
        color_male_idx = i % len(colors_male)
        color_female_idx = i % len(colors_female)
        
        if 'Male' in gender_counts.columns:
            ax = sns.kdeplot(
                gender_counts['Male'],
                label=f"{labels[i]} - Male",
                linestyle=line_styles[style_idx],
                color=colors_male[color_male_idx],
                fill=False,
                alpha=1,
                ax=ax
            )
        
        if 'Female' in gender_counts.columns:
            ax = sns.kdeplot(
                gender_counts['Female'],
                label=f"{labels[i]} - Female",
                linestyle=line_styles[style_idx],
                color=colors_female[color_female_idx],
                fill=False,
                alpha=1,
                ax=ax
            )

    # Title and axis labels
    ax.set_title(title or 'Distribution of Gender Counts Across Multiple Simulations', weight='bold')
    ax.set_xlabel('Count')
    ax.set_ylabel('Density')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-2, 22)
    ax.tick_params(axis='both', labelsize=12)
    
    # Toggle legend visibility
    if show_legend:
        ax.legend(loc='upper right')
    
    return ax
