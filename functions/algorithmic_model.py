# -------------------------------------------------------------------------------
# import packages
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from fairlearn.reductions import ExponentiatedGradient, DemographicParity, TruePositiveRateParity, EqualizedOdds
import numpy as np
import pandas as pd

# -------------------------------------------------------------------------------
# Post Selection Bias Function
def weighted_sampling(X_test_sub, beta, target_gender, n_samples=50):
    """
    Perform weighted sampling based on qualification and gender-based bias.
    
    Parameters:
    - X_test_sub: DataFrame with at least 'qual' and 'gender' columns
    - beta: float, strength of gender bias
    - target_gender: gender to favor (e.g., 0 for male, 1 for female)
    - n_samples: number of individuals to sample
    
    Returns:
    - sampled DataFrame of selected individuals
    """

    # Compute weights
    weights = X_test_sub['qual'] * (1 + beta * (X_test_sub['gender'] == target_gender))

    # Normalize weights (important!)
    weights = weights / weights.sum()

    # Sample according to weights
    selected_indices = np.random.choice(X_test_sub.index, size=n_samples, replace=False, p=weights)

    # Return the selected individuals
    return X_test_sub.loc[selected_indices]

# -------------------------------------------------------------------------------
# function for simulation of the model
def simulation_process(X_test, y_test, model, num_iterations, model_type, beta):

    # list for storing results of the simulation process
    gender_shares_first_stage = []
    gender_shares_second_stage = []
    all_selected_candidates = []
    
    for i in range(num_iterations):

        """First Stage of the Simulation"""

        # make subsample of 80% of test set
        X_test_sub = X_test.sample(frac=0.8)
        y_test_sub = y_test.loc[X_test_sub.index]

        # Make predictions
        if model_type == "logistic":
            y_pred = model.predict(X_test_sub)
        if model_type == "logistic_fair":
            y_pred = model.predict(X_test_sub)
        if model_type == "random_forest":
            y_pred = model.predict(X_test_sub)
        if model_type == "random_forest_fair":
        # 1. Save a copy to use later
            X_test_test = X_test_sub.copy()

            # 2. Extract sensitive attribute for fairness
            protected_attribute_column_test = X_test_test["gender"]

            # 3. Drop it for model input
            X_test_test = X_test_test.drop(columns=["gender"])

            # 4. Predict using fairness-aware model
            y_pred = model.predict(X_test_test, sensitive_features=protected_attribute_column_test)

        # Add predictions to X_test_sub
        X_test_sub = X_test_sub.copy()
        X_test_sub['y_pred'] = y_pred

        # take the ones with y_pred = 1
        X_test_sub = X_test_sub[X_test_sub['y_pred'] == 1]

        # Calculate the normalized counts
        gender_share = X_test_sub['gender'].value_counts(normalize=True)

        # Safely get shares (default to 0 if missing)
        male_share = gender_share.get(0, 0)
        female_share = gender_share.get(1, 0)

        # Calculate percentage point difference
        gender_difference = male_share - female_share

        # store the result
        gender_shares_first_stage.append(gender_difference)

        """Second Stage of the Simulation"""

        # create qualification column
        X_test_sub['qual'] = X_test_sub['ind-university_grade']

        selected_candidates = weighted_sampling(X_test_sub, beta, target_gender=0, n_samples=50)

        selected_candidates['iteration'] = i
        all_selected_candidates.append(selected_candidates)

        gender_share = selected_candidates['gender'].value_counts(normalize=True)

        # Safely get shares (default to 0 if missing)
        male_share = gender_share.get(0, 0)
        female_share = gender_share.get(1, 0)

        # Calculate percentage point difference
        gender_difference = male_share - female_share

        # Store the result
        gender_shares_second_stage.append(gender_difference)

    # Convert the list of DataFrames into a single DataFrame
    all_selected = pd.concat(all_selected_candidates)

    return gender_shares_first_stage, all_selected, gender_shares_second_stage

# -------------------------------------------------------------------------------
# function for logistic regression model
def logistic_regression(X, y, model_type, discrimination, alpha = None, test_size=0.2, random_state=42, 
                        weights = None, enforce_fairness=False, fairness_constraint="demographic_parity"):
    """
    Function to train a logistic regression model, optionally enforcing fairness 
    (demographic parity, equalized odds, or equal opportunity).
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=random_state)

    sensitive_train = X_train['gender']

    base_model = LogisticRegression(max_iter=4000)

    if enforce_fairness:
        if fairness_constraint == "demographic_parity":
            constraint = DemographicParity()
        elif fairness_constraint == "equalized_odds":
            constraint = EqualizedOdds()
        elif fairness_constraint == "equal_opportunity":
            constraint = TruePositiveRateParity()
        else:
            raise ValueError("fairness_constraint must be 'demographic_parity', 'equalized_odds', or 'equal_opportunity'")

        model = ExponentiatedGradient(base_model, constraint)
        model.fit(X_train, y_train, sensitive_features=sensitive_train)
    else:
        model = base_model
        model.fit(X_train, y_train, sample_weight=weights)

    gender_shares1, all_selected, gender_shares2 = simulation_process(X_test, y_test, model, 500, model_type, discrimination)

    return X_train, y_train, model, gender_shares1, all_selected, gender_shares2

# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
# from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
# from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds

# def logistic_regression(X, y, model_type, discrimination, alpha=None, test_size=0.2, random_state=42, 
#                         weights=None, enforce_fairness=False, fairness_constraint="demographic_parity"):
#     """
#     Function to train a logistic regression model, optionally enforcing fairness 
#     (demographic parity, equalized odds, or equal opportunity).
#     """
#     # Split the data into train and test sets
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

#     # Further split the train set into train and validation sets
#     X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=random_state)

#     # Use 'gender' as a sensitive feature in both training and validation sets
#     sensitive_train = X_train['gender']
#     sensitive_val = X_val['gender']

#     base_model = LogisticRegression(max_iter=4000)

#     if enforce_fairness:
#         # Set fairness constraint based on the user input
#         if fairness_constraint == "demographic_parity":
#             constraint = DemographicParity()
#         elif fairness_constraint == "equalized_odds":
#             constraint = EqualizedOdds()
#         else:
#             raise ValueError("fairness_constraint must be 'demographic_parity' or 'equalized_odds'")

#         # Wrap the base model with the fairness constraint
#         model = ExponentiatedGradient(base_model, constraint)
#         model.fit(X_train, y_train, sensitive_features=sensitive_train)
#     else:
#         # Train the model normally without fairness constraints (including 'gender' as a feature)
#         model = base_model
#         model.fit(X_train, y_train, sample_weight=weights)

#     # Evaluate the model on the validation set before using it for simulation
#     y_val_pred = model.predict(X_val)

#     # Performance metrics on the validation set
#     accuracy_val = accuracy_score(y_val, y_val_pred)
#     f1_val = f1_score(y_val, y_val_pred)

#     # Fairness metrics on the validation set
#     if enforce_fairness:
#         if fairness_constraint == "demographic_parity":
#             fairness_metric_val = demographic_parity_difference(y_val, y_val_pred, sensitive_features=X_val['gender'])
#         elif fairness_constraint == "equalized_odds":
#             fairness_metric_val = equalized_odds_difference(y_val, y_val_pred, sensitive_features=X_val['gender'])
#     else:
#         fairness_metric_val = None

#     # Print the performance scores
#     print(f"Validation Accuracy: {accuracy_val:.4f}")
#     print(f"Validation F1 Score: {f1_val:.4f}")
#     if enforce_fairness:
#         print(f"Validation Fairness Metric: {fairness_metric_val:.4f}")
#     else:
#         print("Validation Fairness Metric: None")

#     # Run the simulation process on the test set (including 'gender' as a feature)
#     gender_shares1, all_selected, gender_shares2 = simulation_process(X_test, y_test, model, 500, model_type, discrimination)

#     # Return the variables as requested
#     return X_train, y_train, model, gender_shares1, all_selected, gender_shares2


# -------------------------------------------------------------------------------
# function for a random forest model
from fairlearn.postprocessing import ThresholdOptimizer

def random_forest_model(X, y, model_type, discrimination, test_size=0.2, random_state=42, weights=None, 
                        enforce_fairness=False, fairness_constraint="demographic_parity"):
    """
    Function to train a random forest model, optionally enforcing fairness 
    (demographic parity, equalized odds, or equal opportunity) via postprocessing.
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Define the base model
    base_model = RandomForestClassifier(n_estimators=100, random_state=random_state)

    if enforce_fairness:
        protected_attribute_column = X_train["gender"]
        X_train = X_train.drop(columns=["gender"])

        # Train base model
        base_model.fit(X_train, y_train, sample_weight=weights)

        # Set fairness constraint
        if fairness_constraint == "demographic_parity":
            constraint = "demographic_parity"
        elif fairness_constraint == "equalized_odds":
            constraint = "equalized_odds"
        elif fairness_constraint == "equal_opportunity":
            constraint = "true_positive_rate_parity"
        else:
            raise ValueError("Invalid fairness constraint")

        # Wrap base model with ThresholdOptimizer
        model = ThresholdOptimizer(
            estimator=base_model,
            constraints=constraint,
            prefit=True,
            predict_method="predict_proba"  # very important for models like XGBoost!
        )

        # Fit the ThresholdOptimizer (this actually learns thresholds per group)
        model.fit(X_train, y_train, sensitive_features=protected_attribute_column)

    else:
        # Train the base model normally
        model = base_model
        model.fit(X_train, y_train, sample_weight=weights)

    # Run simulation process
    gender_shares1, all_selected, gender_shares2 = simulation_process(X_test, y_test, model, 500, model_type, discrimination)

    return X_train, y_train, model, gender_shares1, all_selected, gender_shares2