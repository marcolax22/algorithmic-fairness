# -------------------------------------------------------------------------------
# import packages
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.reductions import ExponentiatedGradient, DemographicParity, TruePositiveRateParity, EqualizedOdds
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------------
# Post Selection Bias Function
import numpy as np

def weighted_sampling(X_test_sub, beta, target_gender, n_samples=20, noise_level=1e-2, random_state=None):
    """
    Perform weighted sampling with qualification-based weights and gender bias,
    adding slight noise to make the selection more diverse/random.

    Parameters:
    - beta: float, level of gender-based weighting
    - target_gender: value used to identify the favored group
    - n_samples: number of samples to draw
    - noise_level: small float to inject randomness
    - random_state: int or np.random.Generator for reproducibility
    """
    # Create a random generator
    rng = np.random.default_rng(42)

    # Base weights with gender bias
    base_weights = X_test_sub['qual'] * (1 + beta * (X_test_sub['gender'] == target_gender))

    # Add small noise to break ties and introduce randomness
    noise = rng.normal(loc=1.0, scale=noise_level, size=len(base_weights))
    noisy_weights = base_weights * noise

    # Normalize
    weights = noisy_weights.clip(lower=0)
    weights /= weights.sum()

    # Sample
    selected_indices = rng.choice(X_test_sub.index, size=n_samples, replace=True, p=weights)

    return X_test_sub.loc[selected_indices]


# -------------------------------------------------------------------------------
# function for simulation of the model
def simulation_process(X_test, y_test, model, num_iterations, model_type, beta):

    # list for storing results of the simulation process
    gender_shares_first_stage = []
    gender_shares_second_stage = []
    all_selected_candidates = []
    qualification_female_first_stage = []
    qualification_male_first_stage = []
    qualification_male_second_stage = []
    qualification_female_second_stage = []

    male_shares = []
    female_shares = []
    
    for i in range(num_iterations):

        """First Stage of the Simulation"""

        # make subsample of 80% of test set
        X_test_sub = X_test.sample(frac=0.2, random_state=i)
        # y_test_sub = y_test.loc[X_test_sub.index]

        if model_type == "random_forest_fair" or model_type == "logistic_fair":
            # Drop gender for fairness-aware prediction
            A_test = X_test_sub['gender']
            X_model_input = X_test_sub.drop(columns=['gender'])

            # Predict using the fairness-aware model
            y_pred = model.predict(X_model_input, sensitive_features=A_test)

            # Add prediction to a copy of original data that still has gender
            X_test_sub['y_pred'] = y_pred

            # get qualification for male and female for y_pred == 0
            qualification_male = X_test_sub[(X_test_sub['gender'] == 0) & (X_test_sub['y_pred'] == 1)]['ind-university_grade']
            qualification_female = X_test_sub[(X_test_sub['gender'] == 1) & (X_test_sub['y_pred'] == 1)]['ind-university_grade']
      
            qualification_male_first_stage.append(qualification_male)
            qualification_female_first_stage.append(qualification_female)

            # Select candidates predicted as 1
            X_test_sub = X_test_sub[X_test_sub['y_pred'] == 1]

            # Compute gender share from the selected data
            gender_share = X_test_sub['gender'].value_counts(normalize=True)
            male_share = gender_share.get(0, 0)
            female_share = gender_share.get(1, 0)
            gender_difference = male_share - female_share

            # Store result
            male_shares.append(male_share)
            female_shares.append(female_share)
            gender_shares_first_stage.append(gender_difference)

            # """Second Stage of the Simulation"""

            # create qualification column
            uni_norm = X_test_sub['ind-university_grade'] / 100
            test_norm = X_test_sub['ind-testresult'] / 50
            exp_norm = (X_test_sub['ind-previous_exp'] - 1) / (4 - 1)

            # Combine equally weighted
            X_test_sub['qual'] = (uni_norm + test_norm + exp_norm) / 3

            selected_candidates = weighted_sampling(X_test_sub, beta, target_gender=0, n_samples=20)

            selected_candidates['iteration'] = i

            qualification_male = selected_candidates[(selected_candidates['gender'] == 0)]['ind-university_grade']
            qualification_female = selected_candidates[(selected_candidates['gender'] == 1)]['ind-university_grade']
      
            qualification_male_second_stage.append(qualification_male)
            qualification_female_second_stage.append(qualification_female)

            all_selected_candidates.append(selected_candidates)

            gender_share = selected_candidates['gender'].value_counts(normalize=True)

            # Safely get shares (default to 0 if missing)
            male_share = gender_share.get(0, 0)
            female_share = gender_share.get(1, 0)

            # Calculate percentage point difference
            gender_difference = male_share - female_share

            # Store the result
            gender_shares_second_stage.append(gender_difference)
    
        else: 
            y_pred = model.predict(X_test_sub)

            # Add predictions to X_test_sub
            X_test_sub['y_pred'] = y_pred

            # get qualification for male and female for y_pred == 0
            qualification_male = X_test_sub[(X_test_sub['gender'] == 0) & (X_test_sub['y_pred'] == 1)]['ind-university_grade']
            qualification_female = X_test_sub[(X_test_sub['gender'] == 1) & (X_test_sub['y_pred'] == 1)]['ind-university_grade']
           
            qualification_male_first_stage.append(qualification_male)
            qualification_female_first_stage.append(qualification_female)

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

            selected_candidates = weighted_sampling(X_test_sub, beta, target_gender=0, n_samples=20)

            selected_candidates['iteration'] = i

            qualification_male = selected_candidates[(selected_candidates['gender'] == 0)]['ind-university_grade']
            qualification_female = selected_candidates[(selected_candidates['gender'] == 1)]['ind-university_grade']
      
            qualification_male_second_stage.append(qualification_male)
            qualification_female_second_stage.append(qualification_female)

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

    return (gender_shares_first_stage, all_selected, gender_shares_second_stage, 
            qualification_male_first_stage, qualification_female_first_stage,
            qualification_male_second_stage, qualification_female_second_stage)


# -------------------------------------------------------------------------------
# function for logistic regression model
def logistic_regression(X, y, model_type, discrimination, test_size=0.2, random_state=42, 
                        weights = None, enforce_fairness=False, fairness_constraint="demographic_parity"):
    """
    Function to train a logistic regression model, optionally enforcing fairness 
    (demographic parity, equalized odds, or equal opportunity).
    """
     # Split X and y only
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    base_model = LogisticRegression(
        C=1.0,
        penalty='l2',
        solver='liblinear',
        max_iter=100, 
        class_weight=None,
        random_state=random_state)

    if enforce_fairness:
        if fairness_constraint == "demographic_parity":
            constraint =  "demographic_parity"
        elif fairness_constraint == "equalized_odds":
            constraint = "equalized_odds"
        elif fairness_constraint == "equal_opportunity":
            constraint = "true_positive_rate_parity"
        else:
            raise ValueError("fairness_constraint must be 'demographic_parity', 'equalized_odds', or 'equal_opportunity'")
        
        # Align sensitive attribute A to the train/test splits
        A_train = X_train['gender']
        X_train = X_train.drop(columns=['gender'])

        postprocess = ThresholdOptimizer(
            estimator=base_model,
            constraints= constraint,
            prefit=False
        )

        # Fit the model with the training data
        model = postprocess.fit(X_train, y_train, sensitive_features=A_train)
    else:
        model = base_model
        model.fit(X_train, y_train, sample_weight=weights)
    
    # Call simulation process
    (gender_shares1, all_selected, gender_shares2, qual_male_first_stage, qual_female_first_stage,
     qual_male_second_stage, qual_female_second_stage) = simulation_process(
        X_test, y_test, model, 500, model_type, discrimination
    )

    return (X_train, y_train, model, gender_shares1, all_selected, gender_shares2, 
            qual_male_first_stage, qual_female_first_stage, qual_male_second_stage, qual_female_second_stage)

# -------------------------------------------------------------------------------
# function for random forest model
def random_forest_model(X, y, model_type, discrimination, test_size=0.2, random_state=42, weights=None,
                        enforce_fairness=False, fairness_constraint="demographic_parity"):
    # Split X and y only
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Train base model
    base_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=2,
        bootstrap=False,
        class_weight=None,
        random_state=random_state)


    if enforce_fairness:
        if fairness_constraint == "demographic_parity":
            constraint = "demographic_parity"
        elif fairness_constraint == "equalized_odds":
            constraint = "equalized_odds"
        elif fairness_constraint == "equal_opportunity":
            constraint = "true_positive_rate_parity"
        else:
            raise ValueError("Invalid fairness constraint")

        # Align sensitive attribute A to the train/test splits
        A_train = X_train['gender']
        X_train = X_train.drop(columns=['gender'])

        postprocess = ThresholdOptimizer(
            estimator=base_model,
            constraints= constraint,
            prefit=False
        )

        # Fit the model with the training data
        model = postprocess.fit(X_train, y_train, sensitive_features=A_train)
    else:
        model = base_model
        model = model.fit(X_train, y_train, sample_weight=weights)

    # Call simulation process
    (gender_shares1, all_selected, gender_shares2, qual_male_first_stage, qual_female_first_stage,
     qual_male_second_stage, qual_female_second_stage) = simulation_process(
        X_test, y_test, model, 500, model_type, discrimination
    )

    return (X_train, y_train, model, gender_shares1, all_selected, gender_shares2, 
            qual_male_first_stage, qual_female_first_stage, qual_male_second_stage, qual_female_second_stage)

