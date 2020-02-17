"""
This module contains functions for displaying plot and widgets in Jupyter notebooks.
"""

from ipywidgets import IntProgress, HTML, VBox, widgets
from IPython.display import display
from common import *
import plotly.offline as py
import plotly.io as pio
from plotly.graph_objects import Scatter
from datadict.jupyter import DataDict

pio.templates.default = 'plotly_white'

def player_strength_by_horizon(player_eps: pd.DataFrame, horizon: str, dd: pd.DataFrame, current_gw: int):
    """
    Returns a plotly chart with expected points as the y-axis and cost on the x-axis for a specific time horizon. This chart can be displayed in the Jupyter notebook.

    Args:
        player_eps: The data frame with data to chart.
        horizon: The time horizon of the chart, e.g. Next GW, Next 5 GWs, etc.
        dd: The data dictionary to use for formatting.

    Returns:
        The plotly chart.
    """
    player_eps = player_eps[
        ['Name and Short Team', 'Field Position', 'Current Cost', 'Total Points', 'Minutes Percent', 'News And Date', 'Fixtures Next 5 GWs', 'ICT Index', 'Total Points Consistency']
        + [f'Expected Points {next_gw}'  for next_gw in next_gws]
#        + ['Rel. Fixture Strength ' + next_gw for next_gw in next_gws]
        + (['In Team?'] if 'In Team?' in player_eps.columns.values else [])].copy()
    player_eps_formatted = dd.format(player_eps)
    player_eps['Label'] = (player_eps_formatted['Name and Short Team']
                                      + ', ' + player_eps_formatted['Field Position']
                                      + ', Cost: ' + player_eps_formatted['Current Cost']
                                      + ', Total Points: ' + player_eps_formatted['Total Points']
                                      + '<br>Minutes Percent: ' + player_eps_formatted['Minutes Percent']
                                      + ', Consistency: ' + player_eps_formatted['Total Points Consistency']
                                      + ', ICT: ' + player_eps_formatted['ICT Index']
#                                      + ', Rel. Strength: ' + player_eps_formatted['Rel. Fixture Strength '+horizon+' Calc'] \
                                      + '<br>Next: ' + player_eps_formatted['Fixtures Next 5 GWs']
                                      + '<br>News: ' + player_eps_formatted['News And Date'])

    colors = {'GK': 'rgba(31, 119, 180, 1)', 'DEF': 'rgba(255, 127, 14, 1)', 'MID': 'rgba(44, 160, 44, 1)', 'FWD': 'rgba(214, 39, 40, 1)'}

    data = []
    if 'In Team?' in player_eps.columns.values:
        data += [Scatter(**{
            'x': player_eps[player_eps['In Team?'] == True]['Current Cost'],
            'y': player_eps[player_eps['In Team?'] == True]['Expected Points ' + horizon],
            'mode': 'markers',
            'marker': {
                'size': 15,
                'color': 'white',
                'line': {'width': 1},
            },
            'name': 'In Team',
            'text': player_eps[player_eps['In Team?'] == True]['Label']})]

    data += [Scatter(**{
        'x': player_eps[player_eps['Field Position'] == position]['Current Cost'],
        'y': player_eps[player_eps['Field Position'] == position]['Expected Points '+horizon],
        'name': position,
        'mode': 'markers',
        'marker': {
            'color': colors[position],
        },
        'text': player_eps[player_eps['Field Position'] == position]['Label']
    }) for position in position_by_type.values()]

    return (py.iplot(
        {
            'data': data,
            'layout': {
                'title': f'Expected Points for Game Week {current_gw}',
                'xaxis': {'title': 'Current Cost (lower is better)', 'showspikes': True},
                'yaxis': {'title': f'Expected Points {horizon} (higher is better)', 'showspikes': True},
                'hovermode': 'closest'
            }
        }
    ))



def display_team(team: pd.DataFrame, dd: DataDict, in_team: bool = False) -> widgets.Widget:
    """
    Returns a widget that can be used to show the team and summary stats in a Jupyter notebook.
    Args:
        team: The team data frame.
        dd: The data dictionary to use.
        in_team: whether to show the 'In Team?' column.

    Returns:
        A composite widget.
    """
    team = dd.reorder(team)

    if 'Team Position' in team.columns.values:
        team = team.sort_values('Team Position')

    team['Name'] = team.apply(lambda row: row['Name'] + ' (C)' if row['Captain?'] else row['Name'] + ' (V)' if row['Vice Captain?'] else row['Name'], axis=1)
    team_cols = ['Name', 'Team Short Name'] + (['In Team?'] if in_team else []) + ['Selected?', 'Current Cost', 'Field Position', 'Minutes Percent', 'News And Date', 'Expected Points Next GW', 'Expected Points Next 5 GWs', 'Total Points Consistency']

    parts = []
    parts += [widgets.HTML('<h3>Stats</h3>')]
    parts += [dd.display(summarise_team(team), footer=False, descriptions=False)]
    parts += [widgets.HTML('<h3>Team</h3>')]
    parts += [dd.display(team[team_cols], head=15, excel_file='team.xlsx', index=False)]
    return widgets.VBox(parts)


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
                every = int(size / 200)     # every 0.5%
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
                    label.value = '{name}: {index} / ?'.format(
                        name=name,
                        index=index
                    )
                else:
                    progress.value = index
                    label.value = u'{name}: {index} / {size}'.format(
                        name=name,
                        index=index,
                        size=size
                    )
            yield record
    except:
        progress.bar_style = 'danger'
        raise
    else:
        progress.bar_style = 'success'
        progress.value = index
        label.value = "{name}: {index}".format(
            name=name,
            index=str(index or '?')
        )

