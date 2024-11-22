from data_collectors.combined_data_collection import reversal_df
from predictor_store.predictor_model import run_predictor_model
from analyzers.combined_data_analysis import clean_df
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go


def filter_data(df, market_cap=None, setup=None):
    filtered_df = df.copy()
    if market_cap is not None:  # Check if market_cap is provided
        filtered_df = filtered_df[filtered_df['cap'].isin(market_cap)]
    if setup is not None:  # Check if setup is provided
        filtered_df = filtered_df[filtered_df['setup'].isin(setup)]
    return filtered_df




if __name__ == '__main__':
    cap = None
    setup = None

    # Adjust filtering input: Pass setup only if it is not None
    market_cap_filter = [cap] if cap else None
    setup_filter = [setup] if setup else None

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