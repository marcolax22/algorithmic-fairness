# -------------------------------------------------------------------------------
# import packages
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

# -------------------------------------------------------------------------------
# function for logistic regression model
def logistic_regression(X, y, test_size=0.2, random_state=42, weights=None):
    """
    Function to train a logistic regression model on the given data.
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Train the model
    model = LogisticRegression(max_iter=4000)
    model = model.fit(X_train, y_train, sample_weight=weights)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    report = classification_report(y_test, y_pred)

    # Return the model and the evaluation report
    return X_train, X_test, y_train, y_test, y_pred, report, model

# -------------------------------------------------------------------------------
# function for a random forest model
def random_forest(X, y, test_size=0.2, random_state=42, weights=None):
    """
    Function to train a random forest model on the given data.
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    model = model.fit(X_train, y_train, sample_weight=weights)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    report = classification_report(y_test, y_pred)

    # Return the model and the evaluation report
    return X_train, X_test, y_train, y_test, y_pred, report, model