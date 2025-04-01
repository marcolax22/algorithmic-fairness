# -------------------------------------------------------------------------------
# import packages
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------------------------------------------------
# function for fairness plot of counts between two variables
def fairness_plot(df, var1, var2, figsize=(10, 6)):
    """
    Function to plot counts of two variables
    """
    # plot counts
    df.groupby([var1, var2]).size().unstack().plot(kind='bar', figsize=figsize)
    plt.show()

# function for correlation matrix of dataset
def correlation_matrix(df, figsize=(10, 8)):
    """
    Function to plot a correlation matrix of the dataset with only the lower triangle displayed
    and improved aesthetics.
    """
    # Calculate the correlation matrix
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr, mask=mask, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax, cbar_kws={'shrink': 0.8})
    ax.set_title("Correlation Matrix", fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    # Display the plot
    plt.show()