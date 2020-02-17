"""
This module contains shared helper functions.
"""

import pandas as pd
import numpy as np
import warnings

next_gws = ['Next GW', 'Next 5 GWs', 'GWs To End']
position_by_type = {1: 'GK',  2: 'DEF',  3: 'MID',  4:  'FWD'}

def summarise_team(team: pd.DataFrame) -> pd.DataFrame:
    for gw in next_gws:
        team['Expected Points ' + gw] = team.apply(lambda row: row['Expected Points ' + gw] * max(row['Point Factor'], 1), axis=1)

    aggr = {'Current Cost': 'sum'}
    aggr = {**aggr, **{'Expected Points ' + gw: 'sum' for gw in next_gws}}
    aggr = {**aggr, **{'Total Points Consistency': 'mean'}}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        team_summary = team.agg(aggr).to_frame().T
        team_summary['Team'] = 'All Players'
        team_sel_summary = team[team['Selected?'] == True].agg(aggr).to_frame().T
        team_sel_summary['Team'] = 'Selected Players'

    summary = pd.concat([team_summary, team_sel_summary])
    return summary.set_index('Team')


def validate_df(df: pd.DataFrame, df_name: str, required_columns: list):
    if not set(df.columns) >= set(required_columns):
        raise ValueError(
            f'{df_name} must at least include the following columns: {required_columns},  {list(set(required_columns) - set(df.columns))} are missing. Please ensure the data frame contains these columns.')


def last_or_default(series: pd.Series, default=np.nan):
    if series is None:
        return default

    series = series[~series.isnull()]
    if series.shape[0] == 0:
        return default

    return series.iloc[-1]


def value_or_default(value, default=np.nan):
    return default if value is None or isinstance(value, str) and value == '' or not isinstance(value, str) and np.isnan(value) else value


