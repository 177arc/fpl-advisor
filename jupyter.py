"""
This module contains functions for displaying plot and widgets in Jupyter notebooks.
"""

from ipywidgets import IntProgress, HTML, VBox, widgets
from IPython.display import display
from common import *
import plotly.io as pio
import re
from plotly.graph_objects import Scatter, FigureWidget, Figure, Heatmap
from plotly.graph_objs.layout import Shape
from typing import Callable
import warnings
import asyncio
from time import time
import urllib
import traceback

pio.templates.default = 'plotly_white'

SEL_ALL = 'All'
fdr_color_scale = [[0.0, 'rgb(1, 252, 122)'], [0.33, 'rgb(231, 231, 231)'], [0.66, 'rgb(255, 23, 81)'], [1.0, 'rgb(128, 7, 45)']]


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


def interactive(func: Callable, arg_widgets_map: dict) -> widgets.Widget:
    """
    Calls the given function every time one of the widgets in the given arguments changes. It debounces the changes to avoid too frequent updates.

    Args:
        func: Function to call to create the content that is dependent on the widgets. It will get called every time the value
            of any of the widgets changes. As arguments it is passed the output widget and the values of the widgets as named arguments.
        arg_widgets_map:
            Dictionary with the function's parameter names as keys and the corresponding widgets as values.

    Returns:
        The output widget that will be used the display the content that the function generates.
    """

    @debounce(0.3)
    def on_change(event: dict) -> None:
        fkargs = {}
        for arg, widget in arg_widgets_map.items():
            fkargs[arg] = widget.value

        out.clear_output(wait=True)
        func(out, **fkargs)

    out = widgets.Output()
    for widget in arg_widgets_map.values():
        widget.observe(on_change, 'value')

    on_change({})
    return out


def get_player_stats(player: S, ctx: Context) -> DF:
    next_gws = [list(ctx.next_gw_counts)[0], ctx.def_next_gws, list(ctx.next_gw_counts)[-1]]
    return DF(player).T[[f'Expected Points {next_gw}' for next_gw in next_gws] + ['Current Cost', 'Total Points', 'ICT Index', 'Chance Avail This GW', 'Minutes Played', 'Minutes Percent']]


def player_strength_by_horizon(player_eps: DF, players_gw_eps: DF, horizon: str, position: str, team: str, player: str, ctx: Context):
    """
    Returns a plotly chart with expected points as the y-axis and cost on the x-axis for a specific time horizon. This chart can be displayed in the Jupyter notebook.

    Args:
        player_eps: The data frame with data to chart.
        horizon: The time horizon of the chart, e.g. Next GW, Next 8 GWs, etc.

    Returns:
        The plotly chart.
    """

    def if_in_cols(df: DF, col: str, other):
        return df[col].fillna(other) if col in df.columns else other

    def in_team_trace(player_eps: DF) -> Scatter:
        return Scatter(
            x=player_eps['Current Cost'],
            y=player_eps['Expected Points ' + horizon],
            mode='markers',
            marker={'size': 15, 'color': 'white', 'line': {'width': 1}},
            name='In Team',
            text=player_eps['Label'])

    def position_traces(player_eps: DF, ctx: Context) -> list:
        def trace(player_eps: DF, position: str) -> Scatter:
            return Scatter(
                x=player_eps['Current Cost'],
                y=player_eps['Expected Points ' + horizon],
                name=position,
                mode='markers',
                marker={'color': colors[position], 'opacity': player_eps.pipe(if_in_cols, 'Stats Completeness Percent', 100) / 100},
                text=player_eps['Label'])

        return [trace(player_eps[player_eps['Field Position'] == position], position) for position in ctx.position_by_type.values()]

    def event_capture_trace(player_eps: DF) -> Scatter:
        return Scatter(
            x=player_eps['Current Cost'],
            y=player_eps['Expected Points ' + horizon],
            mode='markers',
            opacity=0,
            marker={'size': 0, 'color': 'white'},
            name='Player',
            text=player_eps['Label'])

    def player_clicked(trace, points, selector):
        message.value = ''

        try:
            player = None
            for ind in points.point_inds:
                player = player_eps.iloc[ind]
                player_code = player_eps.index[ind]

            if player is not None:
                player_gw_eps = players_gw_eps.loc[player_code]
                detail.children = tuple([display_player(player, player_gw_eps, ctx)])
            else:
                detail.children = tuple([])

        # Make sure exceptions are displayed in footer because they are swallowed otherwise.
        except Exception as e:
            message.value = traceback.format_exc()

    def break_text(text: str, max_num_per_line: int, sep=','):
        lines = []

        items = text.split(sep)
        while len(items) > 0:
            lines += [sep.join(items[:max_num_per_line])]
            items = items[max_num_per_line:]

        return (sep + '<br>').join(lines)

    player_eps = player_eps[
        ['Name', 'Name and Short Team', 'Long Name', 'Team Name', 'Field Position', 'Current Cost', 'Total Points', 'Minutes Percent',
         'News And Date', 'ICT Index', 'Minutes Played',
         'Chance Avail This GW', 'Stats Completeness Percent', 'Profile Picture', 'Team Last Updated', 'Player Last Updated']
        + [col for col in player_eps.columns if col.startswith('Expected Points ') or col.startswith('Fixtures ')]
        + (['In Team?'] if 'In Team?' in player_eps.columns else [])].copy()

    pad = 0.5
    min_cost, max_cost = (player_eps['Current Cost'].min() - pad, player_eps['Current Cost'].max() + pad)
    min_eps, max_eps = (player_eps[f'Expected Points {horizon}'].min() - pad, player_eps[f'Expected Points {horizon}'].max() + pad)

    if position != SEL_ALL:
        player_eps = player_eps[lambda df: df['Field Position'] == position]

    if team != SEL_ALL:
        player_eps = player_eps[lambda df: df['Team Name'] == team]

    if player is not None and player != '':
        player_eps = player_eps[lambda df: df['Name'].str.lower().str.contains(player.lower())]

    player_eps_formatted = player_eps.pipe(ctx.dd.format)[player_eps.columns].astype(str)
    player_eps['Label'] = (player_eps_formatted['Name and Short Team']
                           + ', ' + player_eps_formatted['Field Position']
                           + ', Exp. Points: ' + player_eps_formatted[f'Expected Points {horizon}']
                           + ', Cost: ' + player_eps_formatted['Current Cost']
                           + ', Total Points: ' + player_eps_formatted['Total Points']
                           + '<br>Minutes Percent: ' + player_eps_formatted['Minutes Percent']
                           + ', Stats Completeness: ' + player_eps_formatted['Fixtures Played To Fixture'] + f'/{ctx.player_fixtures_look_back}'
                           + ', ICT: ' + player_eps_formatted['ICT Index']
                           + '<br>Next: ' + player_eps_formatted[f'Fixtures Next 8 GWs'].map(lambda v: break_text(v, 4))
                           + '<br>News: ' + player_eps_formatted['News And Date']
                           )

    colors = {'GK': 'rgba(31, 119, 180, 1)',
              'DEF': 'rgba(255, 127, 14, 1)',
              'MID': 'rgba(44, 160, 44, 1)',
              'FWD': 'rgba(214, 39, 40, 1)'}

    traces = []
    if 'In Team?' in player_eps.columns:
        traces += [in_team_trace(player_eps[player_eps['In Team?'] == True])]
    traces += position_traces(player_eps, ctx)
    traces += [event_capture_trace(player_eps)]

    chart = FigureWidget(
        traces,
        layout={
            'title': f'Expected Points for Game Week {ctx.next_gw} for {horizon}',
            'xaxis': dict(title='Current Cost (lower is better)', showspikes=True, range=[min_cost, max_cost]),
            'yaxis': dict(title=f'Expected Points {horizon} (higher is better)', showspikes=True, range=[min_eps, max_eps]),
            'hovermode': 'closest'
        }
    )

    # Register the click event handler. Unfortunately, this cannot be done on the scatter trace itself at creation time for some reason.
    for trace in chart.data:
        trace.on_click(player_clicked, True)

    last_updated = player_eps['Player Last Updated'].min()
    data_quality = widgets.HTML(f'<p><center>Data last updated at {last_updated.strftime("%d %b %Y %H:%M:%S")}</center></p>')

    message = widgets.HTML('<p><center>Click on player for more detail!</center></p>')

    out = widgets.Output()
    detail = widgets.VBox([])
    chart_and_detail = widgets.VBox([message, out, chart, detail, data_quality])

    return chart_and_detail


def add_stats_compl(df: DF, ctx: Context, digits: int = 0) -> DF:
    return (df.assign(**{'Stats Completeness': lambda df: df['Fixtures Played To Fixture']
                      .map(lambda v: ('{:.' + str(digits) + 'f}/{:.0f}').format(v, ctx.player_fixtures_look_back))}))


def display_team(team: DF, ctx: Context, in_team: bool = False) -> widgets.Widget:
    """
    Returns a widget that can be used to show the team and summary stats in a Jupyter notebook.
    Args:
        team: The team data frame.
        ctx: Context data, such as the next game week, the current season, the data dictionary, etc.
        in_team: whether to show the 'In Team?' column.

    Returns:
        A composite widget.
    """
    team = (team
            .pipe(ctx.dd.reorder)
            .assign(**{'Name': lambda df: df.apply(lambda row: row['Name'] + ' (C)' if row['Captain?'] else row['Name'] + ' (V)' if row['Vice Captain?'] else row['Name'], axis=1)})
            .pipe(add_stats_compl, ctx))

    if 'Team Position' in team.columns.values:
        team = team.sort_values('Team Position')

    team_cols = ['Name', 'Team Short Name'] + (['In Team?'] if in_team else []) + ['Selected?', 'Current Cost', 'Field Position', 'Minutes Percent', 'News And Date', 'Expected Points Next GW', f'Expected Points {ctx.def_next_gws}',
                                                                                   'Stats Completeness']

    parts = []
    parts += [widgets.HTML('<h3>Stats</h3>')]
    parts += [ctx.dd.display(summarise_team(team, ctx), footer=False, descriptions=False)]
    parts += [widgets.HTML('<h3>Team</h3>')]
    parts += [ctx.dd.display(team[team_cols], head=15, excel_file='team.xlsx', index=False)]
    return widgets.VBox(parts)


def summarise_team(team: DF, ctx: Context) -> DF:
    """
    Calculates summary stats for the given team data frame, including total cost, expected points for the different time horizons and point consistency.
    Args:
        team: The data frame representing the team.
        ctx: Context data, such as the next game week, the current season, the data dictionary, etc.

    Returns:
        A data frame with the summary stats.
    """

    def calc_aggr(df: DF, aggr: dict) -> DF:
        return (df
                .agg(aggr)
                .to_frame()
                .T
                .pipe(add_stats_compl, ctx, 1)
                .drop(columns=['Fixtures Played To Fixture']))

    for ep_col_name in filter_ep_col_names(team.columns.values):
        team[ep_col_name] = team.apply(lambda row: row[ep_col_name] * max(row['Point Factor'], 1), axis=1)

    next_gws = list(ctx.next_gw_counts)
    aggr = {'Current Cost': 'sum'}
    aggr = {**aggr, **{'Expected Points ' + gw: 'sum' for gw in [next_gws[0], ctx.def_next_gws, next_gws[-1]]}}
    aggr = {**aggr, **{'Fixtures Played To Fixture': 'mean'}}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        team_summary = team.pipe(calc_aggr, aggr)
        team_summary['Team'] = 'All Players'
        team_sel_summary = team[team['Selected?'] == True].pipe(calc_aggr, aggr)
        team_sel_summary['Team'] = 'Selected Players'

    summary = pd.concat([team_summary, team_sel_summary])
    return summary.set_index('Team')


def filter_ep_col_names(col_names: list) -> list:
    """
    Filters forward expected points column names from the given list of column names
    Args:
        col_names: The list of column names.

    Returns:
        A list with forward expected points column names.
    """
    return list(filter(lambda x:
                       re.match(r'Expected Points (Next|GWs)', x),
                       col_names))


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

    record = 0
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


def display_player(player: S, player_gw_eps: DF, ctx: Context) -> widgets.widget:
    def display_header(player: S) -> widgets.widget:
        return widgets.HTML(
            f'<h1>{player["Long Name"]} - {player["Team Name"]} - {player["Field Position"]}</h1>')

    def display_stats(player: S) -> widgets.widget:
        return widgets.VBox([widgets.HTML(
            f'<h3>Stats</h3>'),
            player
                .pipe(get_player_stats, ctx)
                .pipe(ctx.dd.display, footer=False, descriptions=False, index=False)])

    def display_news(player: S) -> widgets.widget:
        return widgets.HTML(value=
                            f'<h3>Availability News</h3>'
                            f'<p>{player["News And Date"]}</p>')

    def display_research(player: S) -> widgets.widget:
        player_query = urllib.parse.quote(f'{player["Long Name"]} FPL')
        team_query = urllib.parse.quote(f'{player["Team Name"]}')
        return widgets.HTML(layout=widgets.Layout(padding='0 0 0 50px'), value=
        ('<style>div.player a {text-decoration-line: underline; margin-right: 20px}</style>'
         '<div class="player">'
         f'<h3>Futher Research</h3>'
         f'<a href="https://www.google.co.uk/search?q={player_query}&ie=UTF-8&tbm=nws" target="_blank">Search Google News for {player["Name"]}<br></a>'
         f'<a href="https://www.google.co.uk/search?q={player_query}&ie=UTF-8" target="_blank">Search Google for {player["Name"]}<br></a>'
         f'<a href="https://www.google.co.uk/search?q={team_query}&ie=UTF-8" target="_blank">Search Google for {player["Team Name"]}</a>'
         '</div>'))

    def display_player(player: S) -> widgets.widget:
        img_url = f'https://resources.premierleague.com/premierleague/photos/players/110x140/p{player["Profile Picture"].split(".")[0]}.png'
        return widgets.HTML(layout=widgets.Layout(padding='0 0 0 50px'), value=f'<img src="{img_url}" />')

    def display_eps(player_gw_eps: DF) -> widgets.widget:
        data = (player_gw_eps
                .reset_index()
                .sort_values(['Season', 'Game Week'])
                .assign(**{'Game Week': lambda df: 'GW ' + df['Game Week'].apply('{:.0f}'.format)})
                .assign(**{'Season Game Week': lambda df: df['Season'] + ', GW ' + df['Game Week']}))

        data_formatted = data.pipe(ctx.dd.format)
        data = data.assign(**{'Label': 'FDR: ' + data_formatted['Fixture Short Name FDR'] + ', '
                                       + 'Rel. Strength: ' + data_formatted['Rel Strength'] + ', '
                                       + 'Cost: ' + data_formatted['Fixture Cost']})

        x_axis = [data['Season'], data['Game Week']]
        # x_axis = data['Season Game Week']

        eps_trace = Scatter(x=x_axis,
                            y=data['Expected Points'],
                            name='Expected Points',
                            line=dict(color='rgb(255, 127, 14)'),
                            mode='lines')

        ftp_trace = Scatter(x=x_axis,
                            y=data['Fixture Total Points'],
                            name='Actual Points',
                            line=dict(color='rgba(44, 160, 44, 0.3)'),
                            mode='lines')

        ftpr_trace = Scatter(x=x_axis,
                             y=data['Rolling Avg Game Points'],
                             name='Rolling Actual Points',
                             line=dict(color='rgb(44, 160, 44)'),
                             line_shape='spline',
                             mode='lines')

        fs_trace = Scatter(x=x_axis,
                           y=data['Expected Points'],
                           name='Rel. Strength',
                           mode='markers',
                           marker=dict(color=(data['Team FDR'].fillna(3)), colorscale=fdr_color_scale),
                           text=data['Label'])

        last_gw = [ctx.current_season, f'GW {ctx.next_gw - 1}']
        first_gw = [ctx.current_season, 'GW 1']
        last_season_gws_color = 'rgb(230, 230, 230)'
        past_gws_color = 'rgb(240, 240, 240)'

        last_season_shape = Shape(type='rect', yref='paper', x0=-6, x1=first_gw, y0=0, y1=1, fillcolor=last_season_gws_color, layer='below', line_width=0, opacity=0.5)
        past_shape = Shape(type='rect', yref='paper', x0=first_gw, x1=last_gw, y0=0, y1=1, fillcolor=past_gws_color, layer='below', line_width=0, opacity=0.5)
        start_shape = Shape(type='line', yref='paper', x0=first_gw, x1=first_gw, y0=0, y1=1, line=dict(width=2, color='DarkGrey'), layer='below')
        current_gw_shape = Shape(type='line', yref='paper', x0=last_gw, x1=last_gw, y0=0, y1=1, line=dict(width=2, color='DarkGrey'), layer='below')

        max_points = max(max(data['Expected Points'].max(), data['Fixture Total Points'].max()) + 1, 15)
        min_points = min(data['Expected Points'].min(), data['Fixture Total Points'].min()) - 1

        layout = dict(
            yaxis=dict(title=f'Points', showspikes=True, range=[min_points, max_points]),
            xaxis=dict(tickfont=dict(size=8)),
            shapes=[last_season_shape, start_shape, past_shape, current_gw_shape],
            hovermode='closest',
            legend=dict(yanchor='top', xanchor='left', x=0, y=1, bgcolor='rgba(0,0,0,0)'),
            height=300,
            margin=dict(l=20, r=0, t=5, b=20, pad=0),
            annotations=[
                dict(x=last_gw, y=0.9, yref='paper', text='Last Game Week', ax=80, ay=0),
                dict(x=first_gw, y=0.9, yref='paper', text='Start of Season', ax=-80, ay=0)
            ]
        )
        return widgets.VBox([widgets.HTML('<h3>Expected Points vs Actual Points</h3>'),
                             FigureWidget([ftp_trace, ftpr_trace, eps_trace, fs_trace], layout=layout)])

    if player is None:
        return None

    player_formatted = DF(player).T.pipe(ctx.dd.format).iloc[0]

    return widgets.VBox([
        display_header(player_formatted),
        display_stats(player),
        widgets.HBox([display_news(player_formatted),
                      display_research(player), display_player(player)]),
        display_eps(player_gw_eps)
    ])


def show_eps_vs_cost(player_gw_next_eps_ext: DF, players_gw_team_eps_ext: DF, teams: DF, ctx: Context):
    # Define the event handlers
    def update_eps_chart(out, horizon, position, team, player) -> None:
        with out:
            display(player_strength_by_horizon(player_gw_next_eps_ext, players_gw_team_eps_ext, horizon, position,
                                           team, player, ctx))

    teams_sel = [SEL_ALL]+list(teams['Team Name'])
    positions_sel = [SEL_ALL]+list(ctx.position_by_type.values())

    # Define the selector controls
    horizon_dropdown = widgets.Dropdown(description='Horizon ', options=ctx.next_gw_counts.keys())
    position_dropdown = widgets.Dropdown(description='Position ', options=positions_sel)
    team_dropdown = widgets.Dropdown(description='Team ', options=teams_sel)
    player_text = widgets.Text(description='Player ', placeholder='Enter player name')

    # Add the change value listeners
    selectors = widgets.HBox([horizon_dropdown, position_dropdown, team_dropdown, player_text])

    out = interactive(update_eps_chart, {'horizon': horizon_dropdown, 'position': position_dropdown, 'team': team_dropdown, 'player': player_text})
    return widgets.VBox([selectors, out])


def get_fdr_chart(fdr_by_team_gw: DF, fdr_labels_by_team_gw: DF, title: str, fixed_scale=False) -> Figure:
    colors = ['rgb(55, 85, 35)', 'rgb(1, 252, 122)', 'rgb(231, 231, 231)', 'rgb(255, 23, 81)', 'rgb(128, 7, 45)']

    scale_min = -1 / 3
    scale_tick = 1 / 3
    if fixed_scale:
        FDR_MIN = 1
        FDR_MAX = 5

        fdr_min = fdr_by_team_gw.min().min()
        fdr_max = fdr_by_team_gw.max().max()

        scale_tick = 1 / (fdr_max - fdr_min)
        scale_min = -(fdr_min - FDR_MIN) * scale_tick
        scale_max = (FDR_MAX - fdr_max) * scale_tick

    fdr_color_scale = []
    scale_val = scale_min
    for color in colors:
        if scale_val >= 0 and scale_val <= 1:
            fdr_color_scale += [[scale_val, color]]
        scale_val += scale_tick

    return Figure(layout=dict(title=title),
                     data=Heatmap(
                         z=fdr_by_team_gw,
                         x=fdr_by_team_gw.columns.values,
                         y=fdr_by_team_gw.index.values,
                         text=fdr_labels_by_team_gw,
                         colorscale=fdr_color_scale,
                         hoverinfo='text',
                         hoverongaps=False))