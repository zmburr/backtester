from data_collectors.combined_data_collection import reversal_df
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

def clean_df(df, analysis_type):
    """
    Cleans the DataFrame by converting columns to their appropriate data types,
    tailored for either breakout or reversal analysis.

    Parameters:
    - df: The DataFrame to be cleaned.
    - analysis_type: A string specifying the type of analysis ('breakout' or 'reversal').

    Returns:
    - The cleaned DataFrame.
    """
    # Convert 'date' column to datetime
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # Convert time_of_event (either breakout or reversal) to datetime and adjust timezone
    time_col = 'time_of_breakout' if analysis_type == 'breakout' else 'time_of_reversal'
    df[time_col] = pd.to_datetime(df[time_col], utc=True).dt.tz_convert('America/New_York')

    if 'time_of_high_price' in df.columns:
        if pd.api.types.is_integer_dtype(df['time_of_high_price']):
            df['time_of_high_price'] = pd.to_datetime(df['time_of_high_price'], errors='coerce',
                                                      unit='s').dt.tz_localize('UTC').dt.tz_convert('America/New_York')
        else:
            df['time_of_high_price'] = pd.to_datetime(df['time_of_high_price'], errors='coerce',
                                                      utc=True).dt.tz_convert('America/New_York')

        # Confirm after all conversions

    # Convert duration to timedelta
    duration_col = 'breakout_duration' if analysis_type == 'breakout' else 'reversal_duration'
    if duration_col in df.columns:
        df[duration_col] = pd.to_timedelta(df[duration_col])

    # Convert specific columns to 'category' data type
    categorical_columns = ['ticker', 'trade_grade'] + (['news_type'] if analysis_type == 'breakout' else ['cap'])
    for col in categorical_columns:
        df[col] = df[col].astype('category')

    # Convert numeric columns to float, excluding specific non-numeric columns
    exclude_cols = categorical_columns + ['date', time_col, duration_col, 'breaks_ath', 'breaks_fifty_two_wk','setup','time_of_high_price','intraday_setup']
    numeric_cols = [col for col in df.columns if col not in exclude_cols]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df






if __name__ == '__main__':
    cleaned_reversal_df = clean_df(reversal_df, 'reversal')
    # Features and target
    # Updated features with additional pct_change columns
    features = [
        'pct_from_10mav', 'pct_from_20mav', 'pct_from_50mav', 'pct_from_200mav',
         'gap_pct', 'pct_change_3', 'pct_change_15', 'pct_change_30'
    ]
    target = 'reversal_open_low_pct'


    # Drop rows with missing values in the features or target
    model_df = cleaned_reversal_df.dropna(subset=features + [target])

    # Train-test split
    X = model_df[features]
    y = model_df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}, R^2: {r2_score(y_test, y_pred):.4f}")
    # Example projection for a new data point
    new_data = {
        'pct_from_10mav': [30.16],
        'pct_from_20mav': [51.66],
        'pct_from_50mav': [99.28],
        'pct_from_200mav': [171.06],
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
        x=['10-day MAV', '20-day MAV', '50-day MAV', '200-day MAV', '3-day Change', '15-day Change', '30-day Change',
           'Predicted Reversal'],
        y=[new_data['pct_from_10mav'][0], new_data['pct_from_20mav'][0], new_data['pct_from_50mav'][0],
           new_data['pct_from_200mav'][0], new_data['pct_change_3'][0], new_data['pct_change_15'][0],
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