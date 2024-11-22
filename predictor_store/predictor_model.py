import logging
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import pandas as pd
from predictor_store.predictor_main import FEATURES, TARGET

def tune_random_forest(df, features=FEATURES, target=TARGET):
    """
    Perform hyperparameter tuning on a Random Forest model.

    Parameters:
    - df: DataFrame containing feature and target columns.
    - features: List of features to use for training the model.
    - target: Target column for prediction.

    Returns:
    - best_model: Best tuned Random Forest model.
    """
    # Drop rows with missing values in the features or target
    model_df = df.dropna(subset=features + [target])

    # Train-test split
    X = model_df[features]
    y = model_df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define parameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Perform grid search
    grid_search = GridSearchCV(
        RandomForestRegressor(random_state=42),
        param_grid,
        scoring='neg_mean_absolute_error',
        cv=3,
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print(f"Best Parameters: {grid_search.best_params_}")

    # Evaluate best model
    y_pred = best_model.predict(X_test)
    print(f"Tuned MAE: {mean_absolute_error(y_test, y_pred):.4f}, R^2: {r2_score(y_test, y_pred):.4f}")

    return best_model