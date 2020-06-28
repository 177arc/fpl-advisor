"""
This module contains shared helper functions.
"""

import pandas as pd
import numpy as np
import warnings

next_gws = ['Next GW', 'Next 5 GWs', 'GWs To End']
position_by_type = {1: 'GK',  2: 'DEF',  3: 'MID',  4:  'FWD'}


def validate_df(df: pd.DataFrame, df_name: str, required_columns: list):
    """
    Validates that the given data frame has at least certain columns.
    Args:
        df: The data frame to be validated.
        df_name: The name of the data frame for the error message.
        required_columns:

    Raises:
        ValueError: Thrown if the validation fails.
    """
    if not set(df.columns) >= set(required_columns):
        raise ValueError(
            f'{df_name} must at least include the following columns: {required_columns},  {list(set(required_columns) - set(df.columns))} are missing. Please ensure the data frame contains these columns.')


def last_or_default(series: pd.Series, default=np.nan):
    """
    Returns the last non-null element of the given series. If non found, returns the default.
    Args:
        series: The series.
        default: The default value.

    Returns:
        Returns the last non-null element of the given series. If non found, returns the default.
    """
    if series is None:
        return default

    series = series[~series.isnull()]
    if series.shape[0] == 0:
        return default

    return series.iloc[-1]


def value_or_default(value, default=np.nan):
    """
    Returns the given value if it is not none. Otherwise returns the default value.

    Args:
        value: The value.
        default: The default.

    Returns:
        Returns the given value if it is not none. Otherwise returns the default value.
    """
    return default if value is None or isinstance(value, str) and value == '' or not isinstance(value, str) and np.isnan(value) else value


