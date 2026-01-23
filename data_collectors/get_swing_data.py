"""
get_swing_data.py - Swing Breakout Volume Data Collector
--------------------------------------------------------
Populates swing_breakout_data.xlsx with comprehensive volume analysis:
- Cumulative volume for premarket, 5min, 10min, 15min, 30min, 1hour
- 30-day trailing averages for each timeframe
- Percentage comparisons to historical averages

Data Sources (with automatic fallback):
1. trillium_queries.py (SHEL DataGateway) - preferred
2. polygon_queries.py (Polygon.io API) - fallback
"""

import os
import sys
import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import trillium_queries first, fallback to polygon if not available
HAS_TRILLIUM = False
HAS_POLYGON = False

try:
    from data_queries.trillium_queries import (
        get_intraday, 
        get_levels_data, 
        adjust_date_to_market,
        _ensure_date_str
    )
    HAS_TRILLIUM = True
    DATA_SOURCE = "trillium"
except ImportError as e:
    logging.warning(f"Trillium queries not available: {e}")
    try:
        from data_queries.polygon_queries import (
            get_intraday,
            get_levels_data,
            adjust_date_to_market
        )
        HAS_POLYGON = True
        DATA_SOURCE = "polygon"
    except ImportError as e2:
        logging.error(f"Neither Trillium nor Polygon queries available: {e2}")
        raise ImportError("No data source available. Please ensure either trillium_queries or polygon_queries is available.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Excel file path
EXCEL_PATH = r"C:\Users\zmbur\Downloads\swing_breakout_data.xlsx"

def convert_date_format(date_str: str) -> str:
    """Convert date from MM/DD/YYYY to YYYY-MM-DD format if needed."""
    try:
        # Handle pandas Timestamp objects
        if hasattr(date_str, 'strftime'):
            return date_str.strftime('%Y-%m-%d')
        
        # Convert string dates
        date_str = str(date_str)
        if "/" in date_str:
            return datetime.strptime(date_str, '%m/%d/%Y').strftime('%Y-%m-%d')
        elif " " in date_str:  # Handle datetime strings like "2023-12-07 00:00:00"
            return date_str.split(" ")[0]
        return date_str
    except Exception as e:
        logger.error(f"Error converting date {date_str}: {e}")
        return str(date_str)

def calculate_volume_metrics(ticker: str, date: str) -> Dict[str, Optional[float]]:
    """
    Calculate comprehensive volume metrics for a given ticker and date.
    
    Returns:
        Dictionary containing volume metrics and percentage comparisons
    """
    try:
        date_iso = convert_date_format(date)
        logger.info(f"Calculating volume metrics for {ticker} on {date_iso} using {DATA_SOURCE}")
        
        # Get intraday minute-by-minute data - adjust API call based on data source
        if HAS_TRILLIUM:
            intraday_df = get_intraday(ticker, date_iso, "bar-1min")
        elif HAS_POLYGON:
            intraday_df = get_intraday(ticker, date_iso, multiplier=1, timespan='minute')
        else:
            raise ValueError("No data source available")
        
        if intraday_df is None or intraday_df.empty:
            logger.warning(f"No intraday data found for {ticker} on {date_iso}")
            return _empty_metrics()
        
        # Calculate current day volumes
        current_volumes = _calculate_current_day_volumes(intraday_df)
        
        # Get 30-day trailing averages
        trailing_averages = _calculate_30day_averages(ticker, date_iso)
        
        # Calculate percentage comparisons
        percentages = _calculate_percentage_comparisons(current_volumes, trailing_averages)
        
        # Combine all metrics
        metrics = {**current_volumes, **trailing_averages, **percentages}
        
        logger.info(f"Successfully calculated metrics for {ticker}")
        return metrics
        
    except Exception as e:
        logger.error(f"Error calculating volume metrics for {ticker} on {date}: {e}")
        return _empty_metrics()

def _calculate_current_day_volumes(intraday_df: pd.DataFrame) -> Dict[str, Optional[float]]:
    """Calculate volume for different time periods on the current day."""
    try:
        # Ensure we have a DatetimeIndex for time filtering
        if not isinstance(intraday_df.index, pd.DatetimeIndex):
            logger.warning("Intraday data doesn't have DatetimeIndex, attempting to convert")
            return {}
        
        volumes = {}
        
        # Premarket volume (6:00 AM - 9:30 AM)
        try:
            premarket = intraday_df.between_time("06:00", "09:29")
            volumes['premarket_vol'] = float(premarket['volume'].sum()) if not premarket.empty else 0.0
        except Exception as e:
            logger.warning(f"Error calculating premarket volume: {e}")
            volumes['premarket_vol'] = None
        
        # Regular session volumes (cumulative from market open)
        try:
            # First 5 minutes (9:30 - 9:35)
            first_5min = intraday_df.between_time("09:30", "09:34")
            volumes['vol_first_5min'] = float(first_5min['volume'].sum()) if not first_5min.empty else 0.0
            
            # First 10 minutes (9:30 - 9:40)
            first_10min = intraday_df.between_time("09:30", "09:39")
            volumes['vol_first_10min'] = float(first_10min['volume'].sum()) if not first_10min.empty else 0.0
            
            # First 15 minutes (9:30 - 9:45)
            first_15min = intraday_df.between_time("09:30", "09:44")
            volumes['vol_first_15min'] = float(first_15min['volume'].sum()) if not first_15min.empty else 0.0
            
            # First 30 minutes (9:30 - 10:00)
            first_30min = intraday_df.between_time("09:30", "09:59")
            volumes['vol_first_30min'] = float(first_30min['volume'].sum()) if not first_30min.empty else 0.0
            
            # First 1 hour (9:30 - 10:30)
            first_1hour = intraday_df.between_time("09:30", "10:29")
            volumes['vol_first_1hour'] = float(first_1hour['volume'].sum()) if not first_1hour.empty else 0.0
            
        except Exception as e:
            logger.warning(f"Error calculating regular session volumes: {e}")
            for key in ['vol_first_5min', 'vol_first_10min', 'vol_first_15min', 'vol_first_30min', 'vol_first_1hour']:
                volumes[key] = None
        
        return volumes
        
    except Exception as e:
        logger.error(f"Error in _calculate_current_day_volumes: {e}")
        return {}

def _calculate_30day_averages(ticker: str, date: str) -> Dict[str, Optional[float]]:
    """Calculate 30-day trailing averages for each time period."""
    try:
        logger.info(f"Calculating 30-day averages for {ticker}")
        
        # Get 30 days of historical data - adjust API call based on data source
        if HAS_TRILLIUM:
            historical_df = get_levels_data(ticker, date, 40, "bar-1day")  # Get more days to ensure we have 30 trading days
        elif HAS_POLYGON:
            historical_df = get_levels_data(ticker, date, 40, 1, 'day')  # Polygon API format
        else:
            raise ValueError("No data source available")
        
        if historical_df is None or historical_df.empty:
            logger.warning(f"No historical data found for {ticker}")
            return _empty_trailing_averages()
        
        # Get the last 30 trading days (excluding current day)
        cutoff_date = pd.to_datetime(date)
        if isinstance(historical_df.index, pd.DatetimeIndex):
            # Handle timezone-aware indexes
            if historical_df.index.tz is not None and cutoff_date.tz is None:
                cutoff_date = cutoff_date.tz_localize(historical_df.index.tz)
            elif historical_df.index.tz is None and cutoff_date.tz is not None:
                cutoff_date = cutoff_date.tz_localize(None)
            historical_df = historical_df[historical_df.index < cutoff_date]
        
        historical_df = historical_df.tail(30)  # Get last 30 days
        
        if len(historical_df) < 10:  # Need reasonable sample size
            logger.warning(f"Insufficient historical data for {ticker}: only {len(historical_df)} days")
            return _empty_trailing_averages()
        
        # Calculate averages for each time period over the 30-day period
        averages = {}
        
        # For each historical day, we need to get intraday data and calculate volumes
        daily_volumes = {
            'premarket': [],
            'first_5min': [],
            'first_10min': [],
            'first_15min': [],
            'first_30min': [],
            'first_1hour': []
        }
        
        # Sample from recent days to estimate averages (to avoid too many API calls)
        sample_days = historical_df.tail(10).index  # Use last 10 days as sample
        
        for sample_date in sample_days:
            try:
                date_str = sample_date.strftime('%Y-%m-%d')
                
                # Get intraday data - adjust API call based on data source
                if HAS_TRILLIUM:
                    day_intraday = get_intraday(ticker, date_str, "bar-1min")
                elif HAS_POLYGON:
                    day_intraday = get_intraday(ticker, date_str, multiplier=1, timespan='minute')
                else:
                    continue
                
                if day_intraday is not None and not day_intraday.empty and isinstance(day_intraday.index, pd.DatetimeIndex):
                    # Calculate volumes for this day
                    premarket = day_intraday.between_time("06:00", "09:29")
                    daily_volumes['premarket'].append(float(premarket['volume'].sum()) if not premarket.empty else 0.0)
                    
                    first_5min = day_intraday.between_time("09:30", "09:34")
                    daily_volumes['first_5min'].append(float(first_5min['volume'].sum()) if not first_5min.empty else 0.0)
                    
                    first_10min = day_intraday.between_time("09:30", "09:39")
                    daily_volumes['first_10min'].append(float(first_10min['volume'].sum()) if not first_10min.empty else 0.0)
                    
                    first_15min = day_intraday.between_time("09:30", "09:44")
                    daily_volumes['first_15min'].append(float(first_15min['volume'].sum()) if not first_15min.empty else 0.0)
                    
                    first_30min = day_intraday.between_time("09:30", "09:59")
                    daily_volumes['first_30min'].append(float(first_30min['volume'].sum()) if not first_30min.empty else 0.0)
                    
                    first_1hour = day_intraday.between_time("09:30", "10:29")
                    daily_volumes['first_1hour'].append(float(first_1hour['volume'].sum()) if not first_1hour.empty else 0.0)
                    
            except Exception as e:
                logger.warning(f"Error getting intraday data for {date_str}: {e}")
                continue
        
        # Calculate averages
        for period, volumes in daily_volumes.items():
            if volumes:
                averages[f'avg_30day_{period}_vol'] = float(np.mean(volumes))
            else:
                averages[f'avg_30day_{period}_vol'] = None
        
        # Also calculate overall daily volume average
        if not historical_df.empty and 'volume' in historical_df.columns:
            averages['avg_30day_daily_vol'] = float(historical_df['volume'].mean())
        else:
            averages['avg_30day_daily_vol'] = None
        
        return averages
        
    except Exception as e:
        logger.error(f"Error calculating 30-day averages: {e}")
        return _empty_trailing_averages()

def _calculate_percentage_comparisons(current_volumes: Dict, trailing_averages: Dict) -> Dict[str, Optional[float]]:
    """Calculate percentage differences between current volumes and trailing averages."""
    try:
        percentages = {}
        
        # Map current volume keys to their corresponding average keys
        volume_mappings = {
            'premarket_vol': 'avg_30day_premarket_vol',
            'vol_first_5min': 'avg_30day_first_5min_vol',
            'vol_first_10min': 'avg_30day_first_10min_vol',
            'vol_first_15min': 'avg_30day_first_15min_vol',
            'vol_first_30min': 'avg_30day_first_30min_vol',
            'vol_first_1hour': 'avg_30day_first_1hour_vol'
        }
        
        for current_key, avg_key in volume_mappings.items():
            current_vol = current_volumes.get(current_key)
            avg_vol = trailing_averages.get(avg_key)
            
            if current_vol is not None and avg_vol is not None and avg_vol > 0:
                pct_diff = ((current_vol - avg_vol) / avg_vol) * 100
                percentages[f'pct_diff_{current_key}'] = float(pct_diff)
            else:
                percentages[f'pct_diff_{current_key}'] = None
        
        return percentages
        
    except Exception as e:
        logger.error(f"Error calculating percentage comparisons: {e}")
        return {}

def _empty_metrics() -> Dict[str, Optional[float]]:
    """Return empty metrics dictionary with all expected keys."""
    return {
        # Current day volumes
        'premarket_vol': None,
        'vol_first_5min': None,
        'vol_first_10min': None,
        'vol_first_15min': None,
        'vol_first_30min': None,
        'vol_first_1hour': None,
        
        # 30-day averages
        **_empty_trailing_averages(),
        
        # Percentage differences
        'pct_diff_premarket_vol': None,
        'pct_diff_vol_first_5min': None,
        'pct_diff_vol_first_10min': None,
        'pct_diff_vol_first_15min': None,
        'pct_diff_vol_first_30min': None,
        'pct_diff_vol_first_1hour': None
    }

def _empty_trailing_averages() -> Dict[str, Optional[float]]:
    """Return empty trailing averages dictionary."""
    return {
        'avg_30day_premarket_vol': None,
        'avg_30day_first_5min_vol': None,
        'avg_30day_first_10min_vol': None,
        'avg_30day_first_15min_vol': None,
        'avg_30day_first_30min_vol': None,
        'avg_30day_first_1hour_vol': None,
        'avg_30day_daily_vol': None
    }

def load_excel_data() -> pd.DataFrame:
    """Load the Excel file with ticker and date data."""
    try:
        if not os.path.exists(EXCEL_PATH):
            raise FileNotFoundError(f"Excel file not found at {EXCEL_PATH}")
        
        df = pd.read_excel(EXCEL_PATH)
        logger.info(f"Loaded Excel file with {len(df)} rows and columns: {list(df.columns)}")
        
        # Ensure required columns exist
        required_cols = ['ticker', 'date']  # Adjust these based on your actual column names
        
        # Try common variations of column names
        column_mappings = {
            'ticker': ['ticker', 'Ticker', 'TICKER', 'symbol', 'Symbol', 'SYMBOL'],
            'date': ['date', 'Date', 'DATE', 'trading_date', 'Date', 'trade_date']
        }
        
        for standard_name, variations in column_mappings.items():
            found = False
            for var in variations:
                if var in df.columns:
                    if var != standard_name:
                        df = df.rename(columns={var: standard_name})
                    found = True
                    break
            if not found:
                raise ValueError(f"Required column '{standard_name}' not found. Available columns: {list(df.columns)}")
        
        # Clean up data
        df = df.dropna(subset=['ticker', 'date'])
        logger.info(f"After cleanup: {len(df)} rows")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading Excel file: {e}")
        raise

def save_excel_data(df: pd.DataFrame) -> None:
    """Save the updated DataFrame back to Excel."""
    try:
        # Create backup
        backup_path = EXCEL_PATH.replace('.xlsx', '_backup.xlsx')
        if os.path.exists(EXCEL_PATH):
            try:
                import shutil
                shutil.copy2(EXCEL_PATH, backup_path)
                logger.info(f"Created backup at {backup_path}")
            except Exception as backup_error:
                logger.warning(f"Could not create backup: {backup_error}")
        
        # Try to save updated data
        try:
            df.to_excel(EXCEL_PATH, index=False)
            logger.info(f"Saved updated data to {EXCEL_PATH}")
        except PermissionError:
            # File is likely open in Excel, save with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            alt_path = EXCEL_PATH.replace('.xlsx', f'_updated_{timestamp}.xlsx')
            df.to_excel(alt_path, index=False)
            logger.warning(f"Original file locked. Saved updated data to {alt_path}")
            print(f"WARNING: Original file was locked. Updated data saved to: {alt_path}")
        
    except Exception as e:
        logger.error(f"Error saving Excel file: {e}")
        raise

def populate_volume_data(df: pd.DataFrame) -> pd.DataFrame:
    """Populate the DataFrame with volume metrics for all rows."""
    logger.info("Starting volume data population...")
    
    # Add all expected columns if they don't exist
    expected_columns = list(_empty_metrics().keys())
    for col in expected_columns:
        if col not in df.columns:
            df[col] = None
    
    # Process each row
    for index, row in df.iterrows():
        ticker = str(row['ticker']).upper()
        date = str(row['date'])
        
        # Check if this row already has data (skip if all volume columns are populated)
        volume_cols = ['premarket_vol', 'vol_first_5min', 'vol_first_10min', 
                      'vol_first_15min', 'vol_first_30min', 'vol_first_1hour']
        
        if all(pd.notna(row[col]) for col in volume_cols if col in row):
            logger.info(f"Skipping {ticker} on {date} - already has volume data")
            continue
        
        logger.info(f"Processing {ticker} on {date} ({index + 1}/{len(df)})")
        
        # Calculate metrics
        metrics = calculate_volume_metrics(ticker, date)
        
        # Update the DataFrame
        for key, value in metrics.items():
            df.at[index, key] = value
    
    logger.info("Finished populating volume data")
    return df

def main():
    """Main function to load Excel data, populate volume metrics, and save results."""
    try:
        logger.info("Starting swing breakout volume data collection...")
        logger.info(f"Using data source: {DATA_SOURCE}")
        
        if HAS_TRILLIUM:
            logger.info("Using Trillium SHEL DataGateway for market data")
        elif HAS_POLYGON:
            logger.info("Using Polygon.io API for market data")
        else:
            raise ValueError("No data source available")
        
        # Load Excel data
        df = load_excel_data()
        
        # Populate volume data
        df = populate_volume_data(df)
        
        # Save updated data
        save_excel_data(df)
        
        # Print summary
        print("\n" + "="*80)
        print("SWING BREAKOUT VOLUME DATA COLLECTION COMPLETE")
        print("="*80)
        print(f"Data Source: {DATA_SOURCE.upper()}")
        print(f"Processed {len(df)} tickers")
        
        # Show sample of results
        volume_cols = ['ticker', 'date', 'premarket_vol', 'vol_first_5min', 'vol_first_30min', 
                      'pct_diff_premarket_vol', 'pct_diff_vol_first_30min']
        available_cols = [col for col in volume_cols if col in df.columns]
        
        if available_cols:
            print(f"\nSample results:")
            print(df[available_cols].head().to_string(index=False))
        
        logger.info("Successfully completed swing breakout volume data collection")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()
