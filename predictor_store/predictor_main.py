from data_collectors.combined_data_collection import reversal_df
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from analyzers.combined_data_analysis import clean_df
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go

# Define a global list of features to make it easy to update them in one place
FEATURES = [
    'pct_from_10mav', 'pct_from_20mav', 'gap_pct',
    'pct_change_3', 'pct_change_15', 'pct_change_30',
    'one_day_before_range_pct', 'two_day_before_range_pct', 'three_day_before_range_pct',
    'percent_of_vol_one_day_before', 'percent_of_vol_two_day_before', 'percent_of_vol_three_day_before'
]
TARGET = 'reversal_open_low_pct'

def filter_data(df, market_cap=None, setup=None):
    filtered_df = df.copy()
    if market_cap is not None:  # Check if market_cap is provided
        filtered_df = filtered_df[filtered_df['cap'].isin(market_cap)]
    if setup is not None:  # Check if setup is provided
        filtered_df = filtered_df[filtered_df['setup'].isin(setup)]
    return filtered_df

def run_predictor_model(df, use_gradient_boosting=False, features=FEATURES, target=TARGET):
    """
    Train and evaluate a regression model on reversal prediction.

    Parameters:
    - df: DataFrame containing feature and target columns.
    - use_gradient_boosting: Whether to use Gradient Boosting instead of Random Forest.
    - features: List of features to use for training the model.
    - target: Target column for prediction.

    Returns:
    - model: Trained regression model.
    - features: List of features used.
    """
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


if __name__ == '__main__':
    cap = None
    setup = None

    # Adjust filtering input: Pass setup only if it is not None
    market_cap_filter = [cap] if cap else None
    setup_filter = [setup] if setup else None

    # Filter and clean data
    filtered_reversal_df = filter_data(reversal_df, market_cap=market_cap_filter, setup=setup_filter)
    cleaned_reversal_df = clean_df(filtered_reversal_df, 'reversal')

    # Train and evaluate the model
    model, features = run_predictor_model(cleaned_reversal_df, use_gradient_boosting=True)

    # Example projection for a new data point
    new_data = {
        'pct_from_10mav': [0.3016],
        'pct_from_20mav': [0.5166],
        'gap_pct': [0.1],
        'pct_change_3': [0.2981],
        'pct_change_15': [0.98],
        'pct_change_30': [1.01],
        'one_day_before_range_pct': [1.8426],
        'two_day_before_range_pct': [1.88126],
        'three_day_before_range_pct': [1.50368],
        'percent_of_vol_one_day_before': [2.1943],
        'percent_of_vol_two_day_before': [1.5897],
        'percent_of_vol_three_day_before': [1.0220],
    }
    new_data_df = pd.DataFrame(new_data)

    # Predict reversal percentage
    predicted_reversal = model.predict(new_data_df)[0]
    print(f"Predicted Reversal: {predicted_reversal:.2f}%")

    # Visualize predictions
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(new_data.keys()) + ['Predicted Reversal'],  # Convert keys to a list
        y=[val[0] for val in new_data.values()] + [predicted_reversal],  # Add predicted value
        name='Reversal Analysis',
        marker_color='blue'
    ))
    fig.update_layout(title="Projected Reversal Percentage with Momentum Features",
                      xaxis_title="Metrics",
                      yaxis_title="Values",
                      template="plotly_dark")
    fig.show()

    # Feature importance
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    print(feature_importance_df)

    # Visualize feature importance
    fig = px.bar(feature_importance_df, x='Importance', y='Feature', orientation='h',
                 title='Feature Importance for Reversal Prediction', template='plotly_dark')
    fig.show()