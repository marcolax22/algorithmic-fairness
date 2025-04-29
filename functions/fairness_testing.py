# -------------------------------------------------------------------------------
# import packages
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve
from abroca import compute_abroca

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
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    return fig 

# -------------------------------------------------------------------------------
# function for ABROCA score
def plot_abroca_curve(y_true, y_pred_proba, sensitive_attr, group_names=("Group 0", "Group 1")):
    # Compute ABROCA score
    abroca_score = compute_abroca(y_true, y_pred_proba, sensitive_attr, compare_type="binary")

    # ROC curves for both groups
    fpr_0, tpr_0, _ = roc_curve(y_true[sensitive_attr == 0], y_pred_proba[sensitive_attr == 0])
    fpr_1, tpr_1, _ = roc_curve(y_true[sensitive_attr == 1], y_pred_proba[sensitive_attr == 1])

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_0, tpr_0, linestyle="--", label=f"{group_names[0]} ROC")
    plt.plot(fpr_1, tpr_1, linestyle="-", label=f"{group_names[1]} ROC")
    plt.fill_between(fpr_0, tpr_0, tpr_1[:len(fpr_0)], color="gray", alpha=0.3, label="ABROCA Region")
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ABROCA Curve\nABROCA Score = {abroca_score:.4f}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return abroca_score
