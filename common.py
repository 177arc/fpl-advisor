"""
This module contains shared helper functions.
"""

import pandas as pd
import numpy as np
from datadict.jupyter import DataDict
import asyncio
from time import time
from ipywidgets import IntProgress, HTML, VBox
from IPython.display import display

# Define type aliases
DF = pd.DataFrame
S = pd.Series


class Context:
    position_by_type: dict = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
    fixture_types: list = ['Home', 'Away', '']
    stats_types: list = ['Goals Scored', 'Goals Conceded', 'Clean Sheets']
    fixture_stats_types: list = [stats_types, fixture_types]

    total_gws: int                      # The number game weeks in a season.
    next_gw: int                        # The upcoming game week.
    def_next_gws: str                   # The default forecast time horizon, e.g. 'Next 8 GWs'
    next_gw_counts: dict                # The map of time horizon to the number of game weeks, e.g. 'Next 8 GWs' is mapped to 8.
    fixtures_look_back: int             # The rolling number of past fixtures to consider when calculating the fixture stats.
    player_fixtures_look_back: int      # The rolling number of past fixtures to consider when calculating the player stats, in particular the expected points.
    last_season: str                    # The name of the last season, e.g. '2019-20'.
    current_season: str                 # The name of the current season, e.g. '2020-21'.
    dd: DataDict                        # The data dictionary to use for column remapping, formatting and descriptions.


def validate_df(df: DF, df_name: str, required_columns: list):
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


def last_or_default(series: S, default=np.nan):
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
    # noinspection PyTypeChecker
    return default if value is None or isinstance(value, str) and value == '' or not isinstance(value, str) and np.isnan(value) else value


class Timer:
    def __init__(self, timeout, callback):
        self._timeout = timeout
        self._callback = callback
        self._task = asyncio.ensure_future(self._job())

    async def _job(self):
        await asyncio.sleep(self._timeout)
        self._callback()

    def cancel(self):
        self._task.cancel()


def throttle(wait: float):
    """
    Decorator that prevents a function from being called more than once every wait period.

    Args:
        wait: Wait period.
    """

    def decorator(fn):
        def throttled(*args, **kwargs):
            nonlocal new_args, new_kwargs, time_of_last_call, scheduled

            def call_it():
                nonlocal new_args, new_kwargs, time_of_last_call, scheduled
                time_of_last_call = time()
                fn(*new_args, **new_kwargs)
                scheduled = False

            time_since_last_call = time() - time_of_last_call
            new_args = args
            new_kwargs = kwargs
            if not scheduled:
                new_wait = max(0, wait - time_since_last_call)
                Timer(new_wait, call_it)
                scheduled = True

        try:
            time_of_last_call = 0
            scheduled = False
            new_args, new_kwargs = None, None
        except:
            print('I')

        return throttled

    return decorator


def debounce(wait: float):
    """
    Decorator that will postpone a function's execution until after `wait` seconds have elapsed since the last time it was invoked.

    Args:
        wait: Wait period.
    """

    def decorator(fn):
        timer = None

        # noinspection PyUnresolvedReferences
        def debounced(*args, **kwargs):
            nonlocal timer

            def call_it():
                fn(*args, **kwargs)

            if timer is not None:
                timer.cancel()
            timer = Timer(wait, call_it)

        return debounced

    return decorator

def get_player_stats(player: S, ctx: Context) -> DF:
    next_gws = [list(ctx.next_gw_counts)[0], ctx.def_next_gws, list(ctx.next_gw_counts)[-1]]
    return DF(player).T[[f'Expected Points {next_gw}' for next_gw in next_gws] + ['Current Cost', 'Total Points', 'ICT Index', 'Chance Avail This GW', 'Minutes Played', 'Minutes Percent']]


def log_progress(sequence, every=None, size=None, name='Items') -> object:
    """
    Shows a progress bar with labels in a Jupyter notebook (see https://github.com/kuk/log-progress).
    Args:
        sequence: The list to iterate over. Each element in the list can cause a progress bar update but the frequency depends on the every parameter.
        every: The frequency of the progress bar update. E.g. update progress bar after every two items.
        size: The number of items in the list.
        name: The description to show.

    Returns:

    """

    is_iterator = False
    if size is None:
        try:
            size = len(sequence)
        except TypeError:
            is_iterator = True
    if size is not None:
        if every is None:
            if size <= 200:
                every = 1
            else:
                every = int(size / 200)  # every 0.5%
    else:
        assert every is not None, 'sequence is iterator, set every'

    if is_iterator:
        progress = IntProgress(min=0, max=1, value=1)
        progress.bar_style = 'info'
    else:
        progress = IntProgress(min=0, max=size, value=0)

    label = HTML()
    box = VBox(children=[label, progress])
    display(box)

    index = 0
    try:
        for index, record in enumerate(sequence, 1):
            if index == 1 or index % every == 0:
                if is_iterator:
                    label.value = f'{name}: {record} / ?'
                else:
                    progress.value = index
                    label.value = f'{name}: {record} / {sequence[-1]}'

            yield record
    except:
        progress.bar_style = 'danger'
        raise
    else:
        progress.bar_style = 'success'
        progress.value = index
        label.value = f'{name}: {record} / {sequence[-1]}'
