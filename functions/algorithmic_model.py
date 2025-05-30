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
from sklearn.metrics import confusion_matrix

# -------------------------------------------------------------------------------
# Post Selection Bias Function
def weighted_sampling(X_test_sub, beta, target_gender, n_samples=20, model = None, noise_level=1e-2, random_state=None):
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

    if model in ["logistic", "random_forest"]:
        base_weights = X_test_sub['qual'] * (1 + beta * (X_test_sub['gender'] == target_gender))
    else:
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
    # This function remains the same as it only uses the test set
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

            selected_candidates = weighted_sampling(X_test_sub, beta, target_gender=0, n_samples=20, model=model_type)

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

            selected_candidates = weighted_sampling(X_test_sub, beta, target_gender=0, n_samples=20, model=model_type)

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
# Modified function for logistic regression model with validation set
def logistic_regression(X, y, model_type, discrimination, test_size=0.2, val_size=0.2, random_state=42, 
                        weights=None, enforce_fairness=False, fairness_constraint="demographic_parity"):
    """
    Function to train a logistic regression model, optionally enforcing fairness 
    (demographic parity, equalized odds, or equal opportunity).
    Now includes a validation set for confusion matrix evaluation.
    """
    from sklearn.model_selection import train_test_split
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from fairlearn.postprocessing import ThresholdOptimizer
    from sklearn.metrics import confusion_matrix
    
    # First split: separate test set from the rest
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Second split: separate validation set from training set
    val_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_adjusted, random_state=random_state
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
            constraint = "demographic_parity"
        elif fairness_constraint == "equalized_odds":
            constraint = "equalized_odds"
        elif fairness_constraint == "equal_opportunity":
            constraint = "true_positive_rate_parity"
        else:
            raise ValueError("fairness_constraint must be 'demographic_parity', 'equalized_odds', or 'equal_opportunity'")
        
        A_train = X_train['gender']
        X_train_no_gender = X_train.drop(columns=['gender'])
        A_val = X_val['gender']
        X_val_no_gender = X_val.drop(columns=['gender'])

        postprocess = ThresholdOptimizer(
            estimator=base_model,
            constraints=constraint,
            prefit=False
        )
        model = postprocess.fit(X_train_no_gender, y_train, sensitive_features=A_train)
        y_val_pred = model.predict(X_val_no_gender, sensitive_features=A_val)
    else:
        model = base_model
        model.fit(X_train, y_train, sample_weight=weights)
        y_val_pred = model.predict(X_val)

    # Validation confusion matrix
    cm = confusion_matrix(y_val, y_val_pred)
    cm_df = pd.DataFrame(cm, index=['Actual 0', 'Actual 1'], columns=['Predicted 0', 'Predicted 1'])

    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    fairness_metrics = {}
    if 'gender' in X_val.columns:
        gender_rates = {}
        for gender_val in [0, 1]:  # 0 for male, 1 for female
            gender_mask = X_val['gender'] == gender_val
            if sum(gender_mask) > 0:
                y_val_gender = y_val[gender_mask]
                y_val_pred_gender = y_val_pred[gender_mask]
                cm_gender = confusion_matrix(y_val_gender, y_val_pred_gender)

                if cm_gender.size == 4:
                    tn_g, fp_g, fn_g, tp_g = cm_gender.ravel()

                    tpr = tp_g / (tp_g + fn_g) if (tp_g + fn_g) > 0 else 0
                    fpr = fp_g / (fp_g + tn_g) if (fp_g + tn_g) > 0 else 0
                    fnr = fn_g / (fn_g + tp_g) if (fn_g + tp_g) > 0 else 0
                    tnr = tn_g / (tn_g + fp_g) if (tn_g + fp_g) > 0 else 0

                    gender_rates[gender_val] = {
                        'tpr': tpr,
                        'fpr': fpr,
                        'fnr': fnr,
                        'tnr': tnr,
                        'tp': tp_g,
                        'fp': fp_g,
                        'fn': fn_g,
                        'tn': tn_g
                    }

        if 0 in gender_rates and 1 in gender_rates:
            tpr_diff = abs(gender_rates[0]['tpr'] - gender_rates[1]['tpr'])
            fpr_diff = abs(gender_rates[0]['fpr'] - gender_rates[1]['fpr'])
            fnr_diff = abs(gender_rates[0]['fnr'] - gender_rates[1]['fnr'])
            tnr_diff = abs(gender_rates[0]['tnr'] - gender_rates[1]['tnr'])

            fairness_metrics['tpr_difference'] = tpr_diff
            fairness_metrics['fpr_difference'] = fpr_diff
            fairness_metrics['fnr_difference'] = fnr_diff
            fairness_metrics['tnr_difference'] = tnr_diff
            fairness_metrics['equalized_odds_difference'] = (tpr_diff + fpr_diff) / 2  # standard metric

            fairness_metrics['male_tpr'] = gender_rates[0]['tpr']
            fairness_metrics['female_tpr'] = gender_rates[1]['tpr']
            fairness_metrics['male_fpr'] = gender_rates[0]['fpr']
            fairness_metrics['female_fpr'] = gender_rates[1]['fpr']
            fairness_metrics['male_fnr'] = gender_rates[0]['fnr']
            fairness_metrics['female_fnr'] = gender_rates[1]['fnr']
            fairness_metrics['male_tnr'] = gender_rates[0]['tnr']
            fairness_metrics['female_tnr'] = gender_rates[1]['tnr']


            fairness_metrics['male_cm'] = {
                'tp': gender_rates[0]['tp'],
                'fp': gender_rates[0]['fp'],
                'fn': gender_rates[0]['fn'],
                'tn': gender_rates[0]['tn']
            }
            fairness_metrics['female_cm'] = {
                'tp': gender_rates[1]['tp'],
                'fp': gender_rates[1]['fp'],
                'fn': gender_rates[1]['fn'],
                'tn': gender_rates[1]['tn']
            }

    # Simulation
    (gender_shares1, all_selected, gender_shares2, qual_male_first_stage, qual_female_first_stage,
     qual_male_second_stage, qual_female_second_stage) = simulation_process(
        X_test, y_test, model, 500, model_type, discrimination
    )

    return (X_train, y_train, model, gender_shares1, all_selected, gender_shares2, 
            qual_male_first_stage, qual_female_first_stage, qual_male_second_stage, qual_female_second_stage,
            cm_df, fairness_metrics)


# -------------------------------------------------------------------------------
# Modified function for random forest model with validation set
def random_forest_model(X, y, model_type, discrimination, test_size=0.2, val_size=0.2, random_state=42, weights=None,
                   enforce_fairness=False, fairness_constraint="demographic_parity"):
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import confusion_matrix
    import numpy as np
    import pandas as pd
    from fairlearn.postprocessing import ThresholdOptimizer
    
    # First split: separate test set from the rest
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Second split: separate validation set from training set
    # Adjust validation size to get the right proportion from the remaining data
    val_adjusted = val_size / (1 - test_size)
    # Using the same random_state to ensure reproducibility across all splits
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_adjusted, random_state=random_state
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
        
        # Align sensitive attribute A to the train/val splits
        A_train = X_train['gender']
        X_train_no_gender = X_train.drop(columns=['gender'])
        
        A_val = X_val['gender']
        X_val_no_gender = X_val.drop(columns=['gender'])
        
        postprocess = ThresholdOptimizer(
            estimator=base_model,
            constraints=constraint,
            prefit=False
        )
        # Fit the model with the training data
        model = postprocess.fit(X_train_no_gender, y_train, sensitive_features=A_train)
        
        # Generate predictions on validation set to create confusion matrix
        y_val_pred = model.predict(X_val_no_gender, sensitive_features=A_val)
        
    else:
        # Keep the original behavior - train with gender included when no fairness constraint
        model = base_model
        model = model.fit(X_train, y_train, sample_weight=weights)
        
        # Predict using the validation data as is (with gender)
        y_val_pred = model.predict(X_val)
    
    # Create and save confusion matrix from validation set
    cm = confusion_matrix(y_val, y_val_pred)
    
    # Save confusion matrix to CSV
    cm_df = pd.DataFrame(cm, 
                        index=['Actual 0', 'Actual 1'], 
                        columns=['Predicted 0', 'Predicted 1'])
    
    # Create filename based on parameters
    fairness_str = "with_fairness" if enforce_fairness else "no_fairness"
    
    # Calculate and print metrics on validation set
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Calculate metrics on validation set
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    fairness_metrics = {}
    if 'gender' in X_val.columns:
        gender_rates = {}
        for gender_val in [0, 1]:  # 0 for male, 1 for female
            gender_mask = X_val['gender'] == gender_val
            if sum(gender_mask) > 0:
                y_val_gender = y_val[gender_mask]
                y_val_pred_gender = y_val_pred[gender_mask]
                cm_gender = confusion_matrix(y_val_gender, y_val_pred_gender)

                if cm_gender.size == 4:
                    tn_g, fp_g, fn_g, tp_g = cm_gender.ravel()

                    tpr = tp_g / (tp_g + fn_g) if (tp_g + fn_g) > 0 else 0
                    fpr = fp_g / (fp_g + tn_g) if (fp_g + tn_g) > 0 else 0
                    fnr = fn_g / (fn_g + tp_g) if (fn_g + tp_g) > 0 else 0
                    tnr = tn_g / (tn_g + fp_g) if (tn_g + fp_g) > 0 else 0

                    gender_rates[gender_val] = {
                        'tpr': tpr,
                        'fpr': fpr,
                        'fnr': fnr,
                        'tnr': tnr,
                        'tp': tp_g,
                        'fp': fp_g,
                        'fn': fn_g,
                        'tn': tn_g
                    }

        if 0 in gender_rates and 1 in gender_rates:
            tpr_diff = abs(gender_rates[0]['tpr'] - gender_rates[1]['tpr'])
            fpr_diff = abs(gender_rates[0]['fpr'] - gender_rates[1]['fpr'])
            fnr_diff = abs(gender_rates[0]['fnr'] - gender_rates[1]['fnr'])
            tnr_diff = abs(gender_rates[0]['tnr'] - gender_rates[1]['tnr'])

            fairness_metrics['tpr_difference'] = tpr_diff
            fairness_metrics['fpr_difference'] = fpr_diff
            fairness_metrics['fnr_difference'] = fnr_diff
            fairness_metrics['tnr_difference'] = tnr_diff
            fairness_metrics['equalized_odds_difference'] = (tpr_diff + fpr_diff) / 2  # standard metric

            fairness_metrics['male_tpr'] = gender_rates[0]['tpr']
            fairness_metrics['female_tpr'] = gender_rates[1]['tpr']
            fairness_metrics['male_fpr'] = gender_rates[0]['fpr']
            fairness_metrics['female_fpr'] = gender_rates[1]['fpr']
            fairness_metrics['male_fnr'] = gender_rates[0]['fnr']
            fairness_metrics['female_fnr'] = gender_rates[1]['fnr']
            fairness_metrics['male_tnr'] = gender_rates[0]['tnr']
            fairness_metrics['female_tnr'] = gender_rates[1]['tnr']


            fairness_metrics['male_cm'] = {
                'tp': gender_rates[0]['tp'],
                'fp': gender_rates[0]['fp'],
                'fn': gender_rates[0]['fn'],
                'tn': gender_rates[0]['tn']
            }
            fairness_metrics['female_cm'] = {
                'tp': gender_rates[1]['tp'],
                'fp': gender_rates[1]['fp'],
                'fn': gender_rates[1]['fn'],
                'tn': gender_rates[1]['tn']
            }
    
    # Call simulation process (still using the test set)
    (gender_shares1, all_selected, gender_shares2, qual_male_first_stage, qual_female_first_stage,
     qual_male_second_stage, qual_female_second_stage) = simulation_process(
        X_test, y_test, model, 500, model_type, discrimination
    )
    
    return (X_train, y_train, model, gender_shares1, all_selected, gender_shares2,
            qual_male_first_stage, qual_female_first_stage, qual_male_second_stage, qual_female_second_stage,
            cm_df, fairness_metrics)  # Now returns the fairness metrics as well