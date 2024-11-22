from data_collectors.combined_data_collection import reversal_df
from predictor_store.predictor_model import run_predictor_model
from analyzers.combined_data_analysis import clean_df
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go


def filter_data(df, market_cap=None, setup=None):
    filtered_df = df.copy()
    if market_cap:
        filtered_df = filtered_df[filtered_df['cap'].isin(market_cap)]
    if setup:
        filtered_df = filtered_df[filtered_df['setup'].isin(setup)]
    return filtered_df




if __name__ == '__main__':
    cap = 'Medium'
    setup = '3DGapFade'
    filtered_reversal_df = filter_data(reversal_df, market_cap=[cap], setup=[setup])
    cleaned_reversal_df = clean_df(filtered_reversal_df, 'reversal')
    # Features and target
    model, features = run_predictor_model(cleaned_reversal_df,use_gradient_boosting=True)
    # Example projection for a new data point
    new_data = {
        'pct_from_10mav': [30.16],
        'pct_from_20mav': [51.66],
        'gap_pct': [0.1],
        'pct_change_3': [.2981],
        'pct_change_15': [.98],
        'pct_change_30': [1.01]}
    new_data_df = pd.DataFrame(new_data)

    # Predict reversal percentage
    predicted_reversal = model.predict(new_data_df)[0]
    print(f"Predicted Reversal: {predicted_reversal:.2f}%")

    # Visualize predictions
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=['10-day MAV', '20-day MAV', '3-day Change', '15-day Change', '30-day Change',
           'Predicted Reversal'],
        y=[new_data['pct_from_10mav'][0], new_data['pct_from_20mav'][0],
            new_data['pct_change_3'][0], new_data['pct_change_15'][0],
           new_data['pct_change_30'][0], predicted_reversal],
        name='Reversal Analysis',
        marker_color='blue'
    ))
    fig.update_layout(title="Projected Reversal Percentage with Momentum Features",
                      xaxis_title="Metrics",
                      yaxis_title="Percentage",
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