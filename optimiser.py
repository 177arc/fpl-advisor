# Code derived from https://github.com/Torvaney/fpl-optimiser
import pandas as pd
import numpy as np
import pulp

position_names = ['gk', 'def', 'mid', 'fwd']
position_min_constraints = [1, 3, 3, 1]
position_max_constraints = [1, 5, 5, 3]

def __add_position_dummy(df):
    for p in position_names:
        df['is_' + str(p)] = np.where(df['Field Position'] == p.upper(), int(1), int(0))
    return df


def __add_team_dummy(df):
    for t in df['Player Team ID'].unique():
        df['team_' + str(t).lower()] = np.where(df['Player Team ID'] == t, int(1), int(0))
    return df


def __add_in_team(df, recommended):
    if recommended is None:
        return  df

    df['in_team'] = np.where(df['In Team?'] == True, int(1), int(0))
    return df

def __remove_news_players(df, recommended):
    if recommended is None:
        return df[pd.isnull(df['News And Date'])]
    else:
        return df[pd.isnull(df['News And Date']) | (df['In Team?'] == True)]


def __optimse_squad(players_df: pd.DataFrame, formation: str = '2-5-5-3', budget: float = 100.0,
                    optimise_on: str = 'Total Points', recommend: int = None) -> pd.DataFrame:
    # Filter out those players with news
    season_stats = (
        players_df
            .assign(cost=lambda df: (df['Current Cost']))
            .pipe(__add_position_dummy)
            .pipe(__add_team_dummy)
            .pipe(__add_in_team, recommend)
    )
    n_players = sum(int(i) for i in formation.split('-'))

    players = season_stats.index

    # Initalise the problem
    fpl_problem = pulp.LpProblem('FPL', pulp.LpMaximize)

    # Create a dictionary of pulp variables with keys from names
    x = pulp.LpVariable.dict('x_ % s', players, lowBound=0, upBound=1,
                             cat=pulp.LpInteger)

    # Player score data
    player_points = dict(
        zip(season_stats.index, np.array(season_stats[optimise_on])))

    # objective function
    fpl_problem += sum([player_points[i] * x[i] for i in players])

    # Constraints
    position_constraints = [int(i) for i in formation.split('-')]

    constraints = dict(zip(position_names, position_constraints))
    constraints['total_cost'] = budget
    constraints['team'] = 3

    # Prepare dictionaries for linear equation
    player_cost = dict(zip(season_stats.index, season_stats.cost))
    player_gk = dict(zip(season_stats.index, season_stats.is_gk))
    player_def = dict(zip(season_stats.index, season_stats.is_def))
    player_mid = dict(zip(season_stats.index, season_stats.is_mid))
    player_fwd = dict(zip(season_stats.index, season_stats.is_fwd))

    # Apply the constraints
    fpl_problem += sum([player_cost[i] * x[i] for i in players]) <= float(constraints['total_cost'])
    fpl_problem += sum([player_gk[i] * x[i] for i in players]) == constraints['gk']
    fpl_problem += sum([player_def[i] * x[i] for i in players]) == constraints['def']
    fpl_problem += sum([player_mid[i] * x[i] for i in players]) == constraints['mid']
    fpl_problem += sum([player_fwd[i] * x[i] for i in players]) == constraints['fwd']

    if not recommend is None:
        player_in_team = dict(zip(season_stats.index, season_stats.in_team))
        fpl_problem += sum([player_in_team[i] * x[i] for i in players]) == n_players - recommend

    for t in season_stats['Player Team ID']:
        player_team = dict(
            zip(season_stats.index, season_stats['team_' + str(t)]))
        fpl_problem += sum([player_team[i] * x[i] for i in players]) <= constraints['team']

    # Solve the thing
    fpl_problem.solve()

    # Filter the players data frame using the selected player ids.
    player_ids = [p for p in players if x[p].value() != 0]
    return players_df[players_df.index.isin(player_ids)]


def __optimse_selection(players_df: pd.DataFrame, optimise_on: str = 'Total Points') -> pd.DataFrame:
    # Filter out those players with news
    season_stats = (
        players_df
            .assign(cost=lambda df: (df['Current Cost']))
            .pipe(__add_position_dummy)
            .pipe(__add_team_dummy)
    )
    n_players = 11
    players = season_stats.index

    # Initalise the problem
    fpl_problem = pulp.LpProblem('FPL', pulp.LpMaximize)

    # Create a dictionary of pulp variables with keys from names
    x = pulp.LpVariable.dict('x_ % s', players, lowBound=0, upBound=1,
                             cat=pulp.LpInteger)

    # Player score data
    player_points = dict(
        zip(season_stats.index, np.array(season_stats[optimise_on])))

    # objective function
    fpl_problem += sum([player_points[i] * x[i] for i in players])

    # Constraints
    constraints = dict(zip(['min_'+p for p in position_names]+['max_'+p for p in position_names],
                           position_min_constraints+position_max_constraints))

    constraints['team'] = 3

    # Prepare dictionaries for linear equation
    player_gk = dict(zip(season_stats.index, season_stats.is_gk))
    player_def = dict(zip(season_stats.index, season_stats.is_def))
    player_mid = dict(zip(season_stats.index, season_stats.is_mid))
    player_fwd = dict(zip(season_stats.index, season_stats.is_fwd))

    # Apply the constraints
    fpl_problem += sum([player_gk[i] * x[i] for i in players]) >= constraints['min_gk']
    fpl_problem += sum([player_gk[i] * x[i] for i in players]) <= constraints['max_gk']
    fpl_problem += sum([player_def[i] * x[i] for i in players]) >= constraints['min_def']
    fpl_problem += sum([player_def[i] * x[i] for i in players]) <= constraints['max_def']
    fpl_problem += sum([player_mid[i] * x[i] for i in players]) >= constraints['min_mid']
    fpl_problem += sum([player_mid[i] * x[i] for i in players]) <= constraints['max_mid']
    fpl_problem += sum([player_fwd[i] * x[i] for i in players]) >= constraints['min_fwd']
    fpl_problem += sum([player_fwd[i] * x[i] for i in players]) <= constraints['max_fwd']
    fpl_problem += sum([x[i] for i in players]) == n_players

    for t in season_stats['Player Team ID']:
        player_team = dict(
            zip(season_stats.index, season_stats['team_' + str(t)]))
        fpl_problem += sum([player_team[i] * x[i] for i in players]) <= constraints['team']

    # Solve the thing
    fpl_problem.solve()

    # Filter the players data frame using the selected player ids.
    player_ids = [p for p in players if x[p].value() != 0]
    return players_df[players_df.index.isin(player_ids)]


def get_optimal_squad(players_df: pd.DataFrame, formation: str = '2-5-5-3', budget: float = 100.0,
                      optimise_team_on: str = 'Total Points', optimise_sel_on: str = None, recommend: int = None) -> pd.DataFrame:
    """
    For the given formation and player data frame tries to maximise the total in the given column to optimise om so that
    the current cost is with the given budget. It also adds the ``Captain/Vice Captain`` column to indicate whether a specific player
    should be captain or vice captain.

    Args:
        players_df: The player data frame with at least the following columns: ``Current Cost``, ``News And Date``, ``Field Position``, ``Player Team ID`` and the column specified in ``optimise_on``. If ``recommend`` is specified, the ``Selected?`` column must also be specified.
        formation: The formation to optimise, e.g, 2-5-5-3 means 2 goal keepers, 5 defender, 5 mid fielders and 3 forwards.
        budget: The budget that the sum of the selected player's ``Current Cost`` should not exceed.
        optimise_team_on: The column in ``players_df`` to use to optimise the team composition, e.g. ``Expected Points Next 5 GWs``. This is different to ``optimise_sel_on`` because the match team selection should be based on the short team where as the team composition should be based on the long term.
        optimise_sel_on: The column in ``players_df`` to use to optimise the team selection.
        recommend: The number of players to recommend transfers for. If specified, ``players_df`` has to include ``Selected?`` to indicate the current squad selection.

    Returns:
        The players from ``players_df`` with the highest value in the ``optimise_on`` column for the given formation that is within the budget.
    """
    required_columns = ['Current Cost', 'News And Date', 'Field Position', 'Player Team ID', optimise_team_on, optimise_sel_on] \
        + ['In Team?'] if not recommend is None else [] \
        + [optimise_sel_on] if not optimise_sel_on is None else []

    if not set(players_df.columns) >= set(required_columns):
        raise ValueError(
            f'players_df must at least include the following columns: {required_columns},  {list(set(required_columns) - set(players_df.columns))} are missing. Please ensure the data frame contains these columns.')

    # Select the best team first
    optimised_team = __optimse_squad(players_df, formation, budget, optimise_team_on, recommend)

    # Check whether a valid solution could be found
    if optimised_team['Current Cost'].sum() > budget+0.01:
        raise Exception(
            f'An optimal solution within the budget {budget:.1f} could not be found. The minimum budget is {optimised_team["Current Cost"].sum()} You need to increase the budget.')

    if not optimise_sel_on is None:
        # Then select the best players from the team to play, i.e. the best valid formation
        selected_team = __optimse_selection(optimised_team, optimise_sel_on)
        optimised_team['Selected?'] = optimised_team.index.map(lambda x: x in selected_team.index.values)

        # Add the result to the output data frame
        optimised_team = optimised_team.sort_values(['Selected?', optimise_sel_on], ascending=False)
        optimised_team['Captain?'] = False
        optimised_team['Vice Captain?'] = False
        optimised_team['Captain?'].iloc[0] = True
        if (optimised_team.shape[0] > 1): optimised_team['Vice Captain?'].iloc[1] = True
        optimised_team['Point Factor'] = optimised_team.apply(lambda row: 0 if not row['Selected?'] else 1 if not row['Captain?'] else 2, axis=1)
        optimised_team = optimised_team.sort_values(['Field Position', optimise_team_on])

    return optimised_team
