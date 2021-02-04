from common import *
from ipywidgets import HTML, HBox, VBox, Widget, Output, Layout, Dropdown, Text
import plotly.io as pio
from plotly.graph_objects import Scatter, FigureWidget
from plotly.graph_objs.layout import Shape
from IPython.display import display
from typing import Callable
import traceback
import urllib

pio.templates.default = 'plotly_white'

SEL_ALL = 'All'
MAX_POINT_SIZE = 15
MIN_POINT_SIZE = 5
MIN_OPACITY = 0.4
POSITION_COLORS = {'GK': 'rgba(31, 119, 180, 1)', 'DEF': 'rgba(255, 127, 14, 1)', 'MID': 'rgba(44, 160, 44, 1)', 'FWD': 'rgba(214, 39, 40, 1)'}
FDR_COLOR_SCALE = [[0.0, 'rgb(1, 252, 122)'], [0.33, 'rgb(231, 231, 231)'], [0.66, 'rgb(255, 23, 81)'], [1.0, 'rgb(128, 7, 45)']]

def interactive(func: Callable, arg_widgets_map: dict) -> Widget:
    """
    Calls the given function every time one of the widgets in the given arguments changes. It debounces the changes to avoid too frequent updates.

    Args:
        func: Function to call to create the content that is dependent on the  It will get called every time the value
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

    out = Output()
    for widget in arg_widgets_map.values():
        widget.observe(on_change, 'value')

    on_change({})
    return out


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
            marker={'size': MAX_POINT_SIZE, 'color': 'white', 'line': {'width': 1}},
            name='In Team',
            text=player_eps['Label'])

    def position_traces(player_eps: DF, ctx: Context) -> list:
        def trace(player_eps: DF, position: str) -> Scatter:
            return Scatter(
                x=player_eps['Current Cost'],
                y=player_eps['Expected Points ' + horizon],
                name=position,
                mode='markers',
                marker={'color': POSITION_COLORS[position],
                        'opacity': player_eps['Opacity'],
                        'size': player_eps['Size']},
                text=player_eps['Label'])

        return [trace(player_eps[player_eps['Field Position'] == position], position) for position in ctx.position_by_type.values()]

    def event_capture_trace(player_eps: DF) -> Scatter:
        return Scatter(
            x=player_eps['Current Cost'],
            y=player_eps['Expected Points ' + horizon],
            mode='markers',
            opacity=0,
            marker={'size': player_eps['Size'], 'color': 'white', 'opacity': 1.0},
            name='Player',
            text=player_eps['Label'])

    def player_clicked(trace, points, selector):
        message.value = ''

        try:
            player = None
            player_code = None
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

    def get_player_eps_chart(player_eps: DF) -> DF:
        return (player_eps[
                    ['Name', 'Name and Short Team', 'Long Name', 'Team Name', 'Field Position', 'Current Cost', 'Total Points', 'Minutes Percent',
                     'News And Date', 'ICT Index', 'Minutes Played', 'Threat To Fixture', 'Influence To Fixture',
                     'Chance Avail This GW', 'Stats Completeness Percent', 'Profile Picture', 'Team Last Updated', 'Player Last Updated']
                    + [col for col in player_eps.columns if col.startswith('Expected Points ') or col.startswith('Fixtures ')]
                    + (['In Team?'] if 'In Team?' in player_eps.columns else [])]
                # Add visualisation columns that need to be calculated on the unfiltered set.
                .assign(**{'Opacity': lambda df: if_in_cols(df, 'Stats Completeness Percent', 100) * (1 - MIN_OPACITY) / 100 + MIN_OPACITY})
                .assign(**{'Size': lambda df: np.maximum(np.where(df['Field Position'] != 'GK',
                                                                  df.groupby('Field Position')['Threat To Fixture'].transform(lambda x: x / x.max() * MAX_POINT_SIZE),
                                                                  df['Influence To Fixture'] / df['Influence To Fixture'].max() * MAX_POINT_SIZE), MIN_POINT_SIZE)}))

    def get_labels(player_eps_chart: DF) -> S:
        player_eps_formatted = player_eps_chart.pipe(ctx.dd.format)[player_eps_chart.columns].astype(str)

        return (player_eps_formatted['Name and Short Team']
                + ', ' + player_eps_formatted['Field Position']
                + ', Cost: ' + player_eps_formatted['Current Cost']
                + '<br>Exp. Points: ' + player_eps_formatted[f'Expected Points {horizon}']
                + ', Total Points: ' + player_eps_formatted['Total Points']
                + '<br>Minutes Percent: ' + player_eps_formatted['Minutes Percent']
                + ', Stats Completeness (Recent Fixtures): ' + player_eps_formatted['Fixtures Played To Fixture'] + f'/{ctx.player_fixtures_look_back}'
                + '<br>Threat (Recent Fixtures): ' + player_eps_formatted['Threat To Fixture']
                + ', ICT: ' + player_eps_formatted['ICT Index']
                + '<br>Next: ' + player_eps_formatted[f'Fixtures Next 8 GWs'].map(lambda v: break_text(v, 4))
                + '<br>News: ' + player_eps_formatted['News And Date']
                )

    player_eps_chart = player_eps.pipe(get_player_eps_chart)

    pad = 0.5
    min_cost, max_cost = (player_eps_chart['Current Cost'].min() - pad, player_eps_chart['Current Cost'].max() + pad)
    min_eps, max_eps = (player_eps_chart[f'Expected Points {horizon}'].min() - pad, player_eps_chart[f'Expected Points {horizon}'].max() + pad)
    last_updated = player_eps_chart['Player Last Updated'].min()

    if position != SEL_ALL:
        player_eps_chart = player_eps_chart[lambda df: df['Field Position'] == position]

    if team != SEL_ALL:
        player_eps_chart = player_eps_chart[lambda df: df['Team Name'] == team]

    if player is not None and player != '':
        player_eps_chart = player_eps_chart[lambda df: df['Name'].str.lower().str.contains(player.lower())]

    # Add labels
    player_eps_chart = player_eps_chart.assign(**{'Label': lambda df: df.pipe(get_labels)})

    traces = []
    if 'In Team?' in player_eps_chart.columns:
        traces += [in_team_trace(player_eps_chart[lambda df: df['In Team?'] == True])]
    traces += position_traces(player_eps_chart, ctx)
    traces += [event_capture_trace(player_eps_chart)]

    chart = FigureWidget(
        traces,
        layout={
            'title': f'Expected Points from Game Week {ctx.next_gw} for {horizon}' if horizon != 'Next GW' else f'Expected Points for Game Week {ctx.next_gw}' ,
            'xaxis': dict(title='Current Cost (lower is better)', showspikes=True, range=[min_cost, max_cost]),
            'yaxis': dict(title=f'Expected Points {horizon} (higher is better)', showspikes=True, range=[min_eps, max_eps]),
            'hovermode': 'closest',
            'legend': dict(itemsizing='constant')
        },
    )

    # Register the click event handler. Unfortunately, this cannot be done on the scatter trace itself at creation time for some reason.
    for trace in chart.data:
        trace.on_click(player_clicked, True)

    data_quality = HTML(f'''<p><center>Color: field position - 
            Size: threat (recent fixtures) for non-goalies, influence (recent fixtures) for goalies - 
            Opacity: stats completeness (recent fixtures)<br> 
            Data last updated: {last_updated.strftime("%d %b %Y %H:%M:%S")}</center></p>''')

    message = HTML('<p><center>Click on player for more detail!</center></p>')

    out = Output()
    detail = VBox([])
    chart_and_detail = VBox([message, out, chart, detail, data_quality])

    return chart_and_detail


def show_eps_vs_cost(player_gw_next_eps_ext: DF, players_gw_team_eps_ext: DF, teams: DF, ctx: Context):
    # Define the event handlers
    def update_eps_chart(out, horizon, position, team, player) -> None:
        with out:
            display(player_strength_by_horizon(player_gw_next_eps_ext, players_gw_team_eps_ext, horizon, position,
                                               team, player, ctx))

    teams_sel = [SEL_ALL] + list(teams['Team Name'])
    positions_sel = [SEL_ALL] + list(ctx.position_by_type.values())

    # Define the selector controls
    horizons = [col.replace('Expected Points ', '') for col in player_gw_next_eps_ext.columns if col.startswith('Expected Points Next ')]
    horizons += [col.replace('Expected Points ', '') for col in player_gw_next_eps_ext.columns if col.startswith('Expected Points GW ')]
    horizon_dropdown = Dropdown(description='Horizon ', options=horizons)
    position_dropdown = Dropdown(description='Position ', options=positions_sel)
    team_dropdown = Dropdown(description='Team ', options=teams_sel)
    player_text = Text(description='Player ', placeholder='Enter player name')

    # Add the change value listeners
    selectors = HBox([horizon_dropdown, position_dropdown, team_dropdown, player_text])

    out = interactive(update_eps_chart, {'horizon': horizon_dropdown, 'position': position_dropdown, 'team': team_dropdown, 'player': player_text})
    return VBox([selectors, out])


def display_player(player: S, player_gw_eps: DF, ctx: Context) -> Widget:
    def display_header(player: S) -> Widget:
        return HTML(
            f'<h1>{player["Long Name"]} - {player["Team Name"]} - {player["Field Position"]}</h1>')

    def display_stats(player: S) -> Widget:
        return VBox([HTML(
            f'<h3>Stats</h3>'),
            player
                .pipe(get_player_stats, ctx)
                .pipe(ctx.dd.display, footer=False, descriptions=False, index=False)])

    def display_news(player: S) -> Widget:
        return HTML(value=
                            f'<h3>Availability News</h3>'
                            f'<p>{player["News And Date"]}</p>')

    def display_research(player: S) -> Widget:
        player_query = urllib.parse.quote(f'{player["Long Name"]} FPL')
        team_query = urllib.parse.quote(f'{player["Team Name"]}')
        return HTML(layout=Layout(padding='0 0 0 50px'), value=
        ('<style>div.player a {text-decoration-line: underline; margin-right: 20px}</style>'
         '<div class="player">'
         f'<h3>Further Research</h3>'
         f'<a href="https://www.google.co.uk/search?q={player_query}&ie=UTF-8&tbm=nws" target="_blank">Search Google News for {player["Name"]}<br></a>'
         f'<a href="https://www.google.co.uk/search?q={player_query}&ie=UTF-8" target="_blank">Search Google for {player["Name"]}<br></a>'
         f'<a href="https://www.google.co.uk/search?q={team_query}&ie=UTF-8" target="_blank">Search Google for {player["Team Name"]}</a>'
         '</div>'))

    def display_player(player: S) -> Widget:
        img_url = f'https://resources.premierleague.com/premierleague/photos/players/110x140/p{player["Profile Picture"].split(".")[0]}.png'
        return HTML(layout=Layout(padding='0 0 0 50px'), value=f'<img src="{img_url}" />')

    def display_eps(player_gw_eps: DF) -> Widget:
        data = (player_gw_eps
                .reset_index()
                .sort_values(['Season', 'Game Week']))

        data_formatted = data.pipe(ctx.dd.format)
        data = (data.assign(**{'Label': 'FDR: ' + data_formatted['Fixture Short Name FDR'] + ', '
                                        + 'Rel. Strength: ' + data_formatted['Rel Strength'] + ', '
                                        + 'Cost: ' + data_formatted['Fixture Cost']})
                .assign(**{'Game Week': lambda df: 'GW ' + df['Game Week'].apply('{:.0f}'.format)})
                .assign(**{'Season Game Week': lambda df: df['Season'] + ', GW ' + df['Game Week']}))

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
                           marker=dict(color=(data['Team FDR'].fillna(3)), colorscale=FDR_COLOR_SCALE),
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
        return VBox([HTML('<h3>Expected Points vs Actual Points</h3>'),
                             FigureWidget([ftp_trace, ftpr_trace, eps_trace, fs_trace], layout=layout)])

    if player is None:
        return None

    player_formatted = DF(player).T.pipe(ctx.dd.format).iloc[0]

    return VBox([
        display_header(player_formatted),
        display_stats(player),
        HBox([display_news(player_formatted),
                      display_research(player), display_player(player)]),
        display_eps(player_gw_eps)
    ])