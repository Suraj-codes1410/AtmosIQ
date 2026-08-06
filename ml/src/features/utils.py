import numpy as np
import pandas as pd
from typing import List


def ensure_chronological(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Sorts dataframe chronologically by date and resets index."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    df[date_col] = df[date_col].dt.strftime("%Y-%m-%d")
    return df


def create_lags(df: pd.DataFrame, cols: List[str], lags: List[int], date_col: str = "date") -> pd.DataFrame:
    """Creates historical lag features for specified columns (strictly shift(k))."""
    df = ensure_chronological(df, date_col)
    for col in cols:
        if col in df.columns:
            for lag in lags:
                lag_col_name = f"{col}_lag_{lag}d"
                df[lag_col_name] = df[col].shift(lag)
    return df


def create_rolling_stats(
    df: pd.DataFrame,
    cols: List[str],
    windows: List[int],
    funcs: List[str] = ["mean", "median", "max", "min", "std", "var"],
    date_col: str = "date"
) -> pd.DataFrame:
    """
    Creates rolling statistics using strictly backward-looking windows.
    For target/pollutant columns, shifts data by 1 first to prevent current-day target leakage.
    """
    df = ensure_chronological(df, date_col)
    
    for col in cols:
        if col in df.columns:
            # Shift by 1 for lag/rolling safety to avoid target leakage of current day
            shifted_series = df[col].shift(1)
            
            for w in windows:
                rolling_obj = shifted_series.rolling(window=w, min_periods=1)
                
                if "mean" in funcs:
                    df[f"{col}_roll_mean_{w}d"] = rolling_obj.mean()
                if "median" in funcs:
                    df[f"{col}_roll_median_{w}d"] = rolling_obj.median()
                if "max" in funcs:
                    df[f"{col}_roll_max_{w}d"] = rolling_obj.max()
                if "min" in funcs:
                    df[f"{col}_roll_min_{w}d"] = rolling_obj.min()
                if "std" in funcs:
                    df[f"{col}_roll_std_{w}d"] = rolling_obj.std().fillna(0.0)
                if "var" in funcs:
                    df[f"{col}_roll_var_{w}d"] = rolling_obj.var().fillna(0.0)
                    
    return df


def deg_to_rad(degrees: pd.Series) -> pd.Series:
    """Converts degrees to radians."""
    return np.deg2rad(degrees)


def clean_inf_and_nans(df: pd.DataFrame) -> pd.DataFrame:
    """Replaces infinity values and performs backward/forward fill for initial lag window NaNs."""
    df = df.replace([np.inf, -np.inf], np.nan)
    # Forward fill then backward fill for initial lag boundary rows
    df = df.ffill().bfill()
    df = df.fillna(0.0)
    return df
