import logging
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import pandas as pd


def run_predictor_model(df, use_gradient_boosting=False):
    """
    Train and evaluate a regression model on reversal prediction.

    Parameters:
    - df: DataFrame containing feature and target columns.
    - use_gradient_boosting: Whether to use Gradient Boosting instead of Random Forest.

    Returns:
    - model: Trained regression model.
    - features: List of features used.
    """
    # Updated features with additional pre-reversal and volume metrics
    features = [
        'pct_from_10mav', 'pct_from_20mav', 'gap_pct',
        'pct_change_3', 'pct_change_15', 'pct_change_30',
        'one_day_before_range_pct', 'two_day_before_range_pct', 'three_day_before_range_pct',
        'percent_of_vol_one_day_before', 'percent_of_vol_two_day_before', 'percent_of_vol_three_day_before'
    ]
    target = 'reversal_open_low_pct'

    # Drop rows with missing values in the features or target
    model_df = df.dropna(subset=features + [target])

    # Train-test split
    X = model_df[features]
    y = model_df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Select model
    if use_gradient_boosting:
        model = GradientBoostingRegressor(random_state=42)
    else:
        model = RandomForestRegressor(random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}, R^2: {r2_score(y_test, y_pred):.4f}")

    # Plot predictions vs actuals
    plt.scatter(y_test, y_pred, alpha=0.5, edgecolors="k")
    plt.xlabel("Actual Reversal")
    plt.ylabel("Predicted Reversal")
    plt.title("Predicted vs Actual Reversals")
    plt.axline((0, 0), slope=1, color="red", linestyle="--", label="Ideal Fit")
    plt.legend()
    plt.show()

    return model, features


def tune_random_forest(df):
    """
    Perform hyperparameter tuning on a Random Forest model.

    Parameters:
    - df: DataFrame containing feature and target columns.

    Returns:
    - best_model: Best tuned Random Forest model.
    """
    features = [
        'pct_from_10mav', 'pct_from_20mav', 'gap_pct',
        'pct_change_3', 'pct_change_15', 'pct_change_30',
        'one_day_before_range_pct', 'two_day_before_range_pct', 'three_day_before_range_pct',
        'percent_of_vol_one_day_before', 'percent_of_vol_two_day_before', 'percent_of_vol_three_day_before'
    ]
    target = 'reversal_open_low_pct'

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