from common import *
from optimiser import get_optimal_team
import warnings
import re
from typing import List, Tuple, Optional
from ipywidgets import HTML, HBox, VBox, Output, Layout, Dropdown, Text, FloatSlider, IntSlider, Label, Widget
from IPython.display import display

def add_stats_compl(df: DF, ctx: Context, digits: int = 0) -> DF:
    return (df.assign(**{'Stats Completeness': lambda df: df['Fixtures Played Recent Fixtures']
                      .map(lambda v: ('{:.' + str(digits) + 'f}/{:.0f}').format(v, ctx.player_fixtures_look_back))}))


def get_expected_points_cols_or_def(expected_points_cols: List[str], ctx: Context) -> List[str]:
    if expected_points_cols is None:
        next_gws = list(ctx.next_gw_counts)
        expected_points_cols = ['Expected Points ' + gw for gw in [next_gws[0], ctx.def_next_gws, next_gws[-1]]]

    # Deduplicate columns
    return list(dict.fromkeys(expected_points_cols))


def display_team(team: DF, current_team: Optional[DF], ctx: Context, in_team: bool = False, expected_points_cols: List[str] = None) -> Widget:
    """
    Returns a widget that can be used to show the team and summary stats in a Jupyter notebook.
    Args:
        team: The team data frame.
        current_team: Data frame representing a team to compare to. This parameter is optional.
        ctx: Context data, such as the next game week, the current season, the data dictionary, etc.
        in_team: Whether to show the 'In Team?' column.
        expected_points_cols: Names of the expected points columns to show for each player and in the summary. If omitted, the expected point columns for the next game week, for the fixed game week horizon and for the end of season horizon are used.

    Returns:
        A composite widget.
    """
    team = (team
            .pipe(ctx.dd.reorder)
            .assign(**{'Name': lambda df: df.apply(lambda row: row['Name'] + ' (C)' if row['Captain?'] else row['Name'] + ' (V)' if row['Vice Captain?'] else row['Name'], axis=1)})
            .pipe(add_stats_compl, ctx))

    expected_points_cols = get_expected_points_cols_or_def(expected_points_cols, ctx)
    team = team.sort_values(['Field Position Code', 'Selected?', expected_points_cols[0]], ascending=[True, False, False])
    is_recommendation = 'Recommendation' in team.columns

    team_cols = (['Name', 'Team Short Name']
                 + (['Recommendation'] if is_recommendation else [])
                 + (['In Team?'] if in_team else [])
                 + ['Selected?', 'Current Cost', 'Field Position', 'Minutes Percent', 'News And Date']
                 + expected_points_cols[0:2]
                 + ['Stats Completeness'])

    team_label = 'New Team' if is_recommendation else 'Team'
    current_team_label = 'Current Team' if is_recommendation else None

    parts = []
    parts += [HTML('<h3>Summary</h3>')]
    parts += [ctx.dd.display(summarise_team(team, team_label, current_team, current_team_label, ctx, expected_points_cols), footer=False, descriptions=False)]

    parts += [HTML('<h3>Team</h3>')]
    parts += [ctx.dd.display(team[team_cols], head=30, excel_file='team.xlsx', index=False)]
    return VBox(parts)


def get_new_team(team: DF):
    """
    Gets the new team from a team with transfer recommendations by removing all out transfer recommendations. If the team
    does not contain any recommendations, it returns the team as is.

    Args:
        team: Data frame to get the new team from.

    Returns:
        Data frame with only actual team players.
    """
    if 'Recommendation' in team.columns:
        return team[lambda df: df['Recommendation'] != 'Transfer out']
    else:
        return team.copy()

def summarise_team(team: DF, team_label: str, other_team: Optional[DF], other_team_label: Optional[str], ctx: Context, expected_points_cols: List[str] = None) -> DF:
    """
    Calculates summary stats for the given team data frame, including total cost, expected points for the different time horizons and point consistency.
    Args:
        team: Data frame representing the team to summarise.
        team_label: Text describing the team. When comparing to another team, this could be 'New Team', otherwise it describes the team and the label could be something like 'Team'.
        other_team: Data frame representing a team to compare to. This parameter is optional.
        other_team_label: Text describing the other team. When comparing to another team, this could be 'Current Team', otherwise this label is ignored.
        ctx: Context data, such as the next game week, the current season, the data dictionary, etc.
        expected_points_cols: Names of the expected points columns to summarise. If omitted, the expected point columns for the next game week, for the fixed game week horizon and for the end of season horizon are used.

    Returns:
        A data frame with the summary stats.
    """

    def calc_aggr(df: DF, aggr: dict) -> DF:
        return (df
                .agg(aggr)
                .to_frame()
                .T
                .pipe(add_stats_compl, ctx, 1)
                .drop(columns=['Fixtures Played Recent Fixtures']))

    expected_points_cols = get_expected_points_cols_or_def(expected_points_cols, ctx)

    for ep_col_name in filter_ep_col_names(team.columns.values):
        team[ep_col_name] = team.apply(lambda row: row[ep_col_name] * max(row['Point Factor'], 1), axis=1)

    aggr = {'Current Cost': 'sum'}
    aggr = {**aggr, **{col: 'sum' for col in expected_points_cols}}
    aggr = {**aggr, **{'Fixtures Played Recent Fixtures': 'mean'}}

    team_summary = DF()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        team = get_new_team(team)
        team_summary = team_summary.append(team
                                           .pipe(calc_aggr, aggr)
                                           .assign(**{'Team': f'{team_label} (all players)'}))
        team_summary = team_summary.append(team[lambda df: df['Selected?']]
                                            .pipe(calc_aggr, aggr)
                                            .assign(**{'Team': f'{team_label} (selected players)'}))

        if other_team is not None:
            other_team = get_new_team(other_team)
            team_summary = team_summary.append(other_team
                                               .pipe(calc_aggr, aggr)
                                               .assign(**{'Team': f'{other_team_label} (all players)'}))
            team_summary = team_summary.append(other_team[lambda df: df['Selected?']]
                                                .pipe(calc_aggr, aggr)
                                                .assign(**{'Team': f'{other_team_label} (selected players)'}))

    return team_summary.set_index('Team')


def display_optimal_team(players_df: DF, ctx: Context, formation: Tuple[int] = (2, 5, 5, 3), expens_player_count: int = None, budget: float = 100.0,
                         optimise_team_on: str = 'Total Points', optimise_sel_on: str = None, recommend: int = None,
                         include: Tuple[str] = (), exclude: Tuple[str] = (), risk: float = 0.2) -> DF:
    """
    For the given formation and player data frame tries to maximise the total in the given column so that
    the current cost is with the given max_budget. It also adds the ``Captain/Vice Captain`` column to indicate whether a specific player
    should be captain or vice captain.

    Args:
        players_df: Player data frame with at least the following columns: ``Current Cost``, ``News And Date``, ``Field Position``, ``Player Team ID`` and the column specified in ``optimise_on``.
            If ``recommend`` is specified, the ``Selected?`` column must also be specified.
        formation: The formation to optimise, e.g, (2, 5, 5, 3) means 2 goal keepers, 5 defender, 5 mid fielders and 3 forwards.
        expens_player_count: Number of expensive players in a biased, optimised team. The number of players must be equal or less than the number of players in the formation if the formation is specified.
        If not specified, the optimised team will not be biased towards more expensive players. Biasing a team can be useful if you are unlikely to need bench players, e.g. for a free hit.
        budget: Budget that the sum of the team player's ``Current Cost`` should not exceed.
        optimise_team_on: Column in ``players`` to use to optimise the team composition, e.g. ``Expected Points Next 5 GWs``.
            This is different to ``optimise_sel_on`` because the match team selection should be based on the short term where as the team composition should be based on the long term.
        optimise_sel_on: Column in ``players`` to use to optimise the team selection.
        recommend: Number of players to recommend transfers for. If specified, ``players`` has to include ``Selected?`` to indicate the current squad selection.
        include: List of player names that must be in the optimised team.
        exclude: List of player names that must NOT be in the optimised team.
        risk: Amount of risk to take when evaluating the column to optimise on. This number has to be between 0 and 1.
            The lower the numbers, the lower the risk. If risk is 1, the completeness of the data that the values in the optimisation columns is based on
            is completely ignored. If risk is 0, the values in the optimisation columns is multiplied by the completeness of the data.
            If risk is between 0 and 1 the completeness of the data is take into account in a way that is proportionate to the risk.

    Returns:
        The players from ``players`` with the highest value in the ``optimise_on`` column for the given formation that is within the max_budget.
    """
    new_team = (get_optimal_team(players_df,
                                 optimise_team_on=optimise_team_on,  # Name of the column to use for optimising the whole team.
                                 optimise_sel_on=optimise_sel_on,  # Name of the column to use for optimising the selection of the team.
                                 formation=formation,  # Formation of the team in format GKP-DEF-MID-FWD.
                                 expens_player_count=expens_player_count,  # Number of expensive players. A number between 14 and 11 will bias the team towards more expensive players.
                                 budget=budget,  # Maximum max_budget available for optimising the team.
                                 include=include,  # List of player names that must be in the team, e.g. ['De Bruyne', 'Mané']
                                 exclude=exclude,  # List of player names that must NOT be in the team.
                                 recommend=recommend,
                                 risk=risk)  # The amount of risk to take when evaluating the column to optimise on. This number has to be between 0 and 1.
                .pipe(ctx.dd.reorder))

    has_in_team = 'In Team?' in new_team.columns
    current_team = players_df[lambda df: df['In Team?'] == True] if has_in_team else None
    return display_team(new_team, current_team, ctx, has_in_team, [optimise_sel_on, optimise_team_on])


def get_horizons(player_gw_next_eps_ext: DF) -> List[str]:
    return ([col.replace('Expected Points ', '') for col in player_gw_next_eps_ext.columns if col.startswith('Expected Points Next ')]
            + [col.replace('Expected Points ', '') for col in player_gw_next_eps_ext.columns if col.startswith('Expected Points GW ')])


def show_opt_team(player_gw_next_eps_ext: DF, max_budget: float, ctx: Context, def_budget: float = None):
    def get_message(optimise_team_on, optimise_sel_on, budget, expens_player_count, include, exclude, risk, recommend) -> None:
        message = '<span style="margin-right: 30px">'

        if recommend is not None:
            message += f'Recommend <strong>{recommend}</strong> transfer(s) with the highest <strong>{optimise_team_on}</strong> '
        else:
            message += f'Create a team with the highest <strong>{optimise_team_on}</strong> '

        message += f'within the budget of <strong>£{budget:.1f}m</strong> biasing the team towards <strong>{expens_player_count}</strong> more expensive player(s) ' + \
                   f'and taking <strong>{risk:.0%}</strong> risk '

        if len(include) > 0 and len(exclude) == 0:
            message += f'while including <strong>{",".join(include)}</strong> '
        elif len(include) == 0 and len(exclude) > 0:
            message += f'while excluding <strong>{",".join(exclude)}</strong> '
        elif len(include) > 0 and len(exclude) > 0:
            message += f'while including <strong>{",".join(include)}</strong> and excluding <strong>{",".join(exclude)}</strong> '

        message += f'and optimise the selection for <strong>{optimise_sel_on}</strong>.</span>'

        return message

    def get_players(players_text: str) -> Tuple:
        return tuple([v.strip() for v in players_text.split(',')]) if players_text != '' else ()

    def get_args():
        return dict(recommend=(recommend_slider.value if has_in_team else None), optimise_team_on='Expected Points ' + opt_for_dropdown.value,
                    budget=budget_slider.value, optimise_sel_on='Expected Points ' + sel_for_dropdown.value,
                    expens_player_count=expens_player_slider.value, include=get_players(include_text.value),
                    exclude=get_players(exclude_text.value), risk=risk_slider.value)

    def validate_include_exclude(df, include_exclude, include_exclude_text) -> None:
        include_exclude_lower = [v.lower() for v in include_exclude]
        if not set(df['Name'].str.lower().values) >= set(include_exclude_lower):
            invalid_include_exclude = [v for v in include_exclude if v.lower() in (set(include_exclude_lower) - set(df["Name"].str.lower().values))]
            if len(invalid_include_exclude) == 1:
                raise ValueError(f'{invalid_include_exclude[0]} in {include_exclude_text} is not a valid player name.')
            else:
                raise ValueError(f'{", ".join(invalid_include_exclude)} in {include_exclude_text} are not valid player names.')

    @debounce(1)
    def on_value_change(change):
        args = get_args()
        error_html.value = ''
        try:
            validate_include_exclude(player_gw_next_eps_ext, args['include'], 'Include')
            validate_include_exclude(player_gw_next_eps_ext, args['exclude'], 'Exclude')

            message_html.value = get_message(**args)
            optimise_team()
            error_html.value = ''
        except Exception as e:
            error_html.value = f'<div class="alert alert-danger">{e}</div>'

    def optimise_team():
        error_html.value = f'<div class="alert alert-success">Optimising ...</div>'
        with out:
            out.clear_output()
            summary = display_optimal_team(players_df=player_gw_next_eps_ext, ctx=ctx, formation=(2, 5, 5, 3),
                                           **get_args())

            display(summary)

    has_in_team = 'In Team?' in player_gw_next_eps_ext.columns
    horizons = get_horizons(player_gw_next_eps_ext)

    if def_budget is None:
        def_budget = max_budget

    # Define the selector controls
    opt_for_dropdown = Dropdown(description='Optimise for', options=horizons, value=ctx.def_next_gws)
    sel_for_dropdown = Dropdown(description='Select for', options=horizons)
    budget_slider = FloatSlider(value=def_budget, min=0, max=max_budget, step=0.1, description='Budget', continuous_update=True, readout_format='.1f')
    expens_player_label = Label('Expensive players')
    expens_player_slider = IntSlider(value=15, min=1, max=15, step=1, continuous_update=True, readout_format='.0f')
    include_text = Text(description='Include', placeholder='Comma separated player names')
    exclude_text = Text(description='Exclude', placeholder='Comma separated player names')
    risk_slider = FloatSlider(value=0.2, min=0, max=1, step=0.1, description='Risk', continuous_update=True, readout_format='.0%')
    recommend_slider = IntSlider(value=1, min=0, max=11, step=1, description='Recommend', continuous_update=True, readout_format='.0f')

    message_html = HTML()
    error_html = HTML(layout=Layout(flex='1 1 0%', width='auto'))

    hbox_layout = Layout(display='flex', flex_flow='row', align_items='stretch', width='100%')

    hboxes = []
    hboxes += [HBox([opt_for_dropdown, sel_for_dropdown]+([recommend_slider] if has_in_team else []), layout=hbox_layout)]
    hboxes += [HBox([include_text, exclude_text, risk_slider], layout=hbox_layout)]
    hboxes += [HBox([budget_slider, expens_player_label, expens_player_slider], layout=hbox_layout)]

    for hbox in hboxes:
        for widget in hbox.children:
            widget.observe(on_value_change, names='value')

    hboxes += [HBox([message_html])]
    hboxes += [HBox([error_html])]
    out = Output()

    on_value_change(None)

    return VBox(hboxes + [out])


def filter_ep_col_names(col_names: list) -> list:
    """
    Filters forward expected points column names from the given list of column names
    Args:
        col_names: The list of column names.

    Returns:
        A list with forward expected points column names.
    """
    return list(filter(lambda x:
                       re.match(r'Expected Points ', x),
                       col_names))