# Code derived from https://github.com/Torvaney/fpl-optimiser
from functools import reduce

import pandas as pd
import numpy as np
from pulp import LpProblem, lpSum, LpVariable, LpInteger, LpMaximize, LpStatus
from typing import Tuple

# Define type aliases
DF = pd.DataFrame

# Define constants
__POS_NAMES = ['GK', 'DEF', 'MID', 'FWD']
__MIN_SEL_FORMATION = (1, 3, 2, 1)
__MAX_SEL_FORMATION = (1, 5, 5, 3)
__MAX_TEAM_FORMATION = (2, 5, 5, 3)
__LOCAL_COL_PREFIX = '_'


def __remove_temp_cols(df: DF) -> DF:
    return df[[col for col in df.columns if not col.startswith(__LOCAL_COL_PREFIX)]]


def __add_position_dummy(df):
    for p in __POS_NAMES:
        df = df.assign(**{'_'+str(p)+'?': lambda df: np.where(df['Field Position'] == p.upper(), int(1), int(0))})

    return df


def __add_team_dummy(df):
    for t in df['Player Team Code'].unique():
        df = df.assign(**{'_Team ' + str(t).lower(): lambda df: np.where(df['Player Team Code'] == t, int(1), int(0))})

    return df


def __add_in_team(df, recommended):
    if recommended is None:
        return df

    return df.assign(**{'_In Team?': lambda df: np.where(df['In Team?'] == True, int(1), int(0))})


def __remove_news_players(df, recommended):
    if recommended is None:
        return df[pd.isnull(df['News And Date'])]
    else:
        return df[pd.isnull(df['News And Date']) | (df['In Team?'])]


def __add_include_players(df, include):
    include_lower = [v.lower() for v in include]
    return df.assign(**{'_Include?': lambda df: np.where(df['Name'].str.lower().isin(include_lower), int(1), int(0))})


def __add_exclude_players(df, exclude):
    exclude_lower = [v.lower() for v in exclude]
    return df.assign(**{'_Exclude?': lambda df: np.where(df['Name'].str.lower().isin(exclude_lower), int(1), int(0))})


def __optimise_team(players_df: DF, min_formation: Tuple[int] = (2, 5, 5, 3), max_formation: Tuple[int] = (2, 5, 5, 3), player_count: int = None, budget: float = None,
                    optimise_on: str = 'Total Points', recommend: int = None,
                    include: Tuple[str] = (), exclude: Tuple[str] = ()) -> DF:
    # Filter out those players with news
    season_stats = (players_df
                    .pipe(__add_position_dummy)
                    .pipe(__add_team_dummy)
                    .pipe(__add_in_team, recommend)
                    .pipe(__add_include_players, include)
                    .pipe(__add_exclude_players, exclude))

    if player_count is None:
        player_count = sum(max_formation)

    players = season_stats.index

    # Initalise the problem
    fpl_problem = LpProblem('FPL', LpMaximize)

    # Create a dictionary of pulp variables with keys from names
    x = LpVariable.dict('x_ % s', players, lowBound=0, upBound=1,
                        cat=LpInteger)

    # Player score data
    player_points = dict(
        zip(season_stats.index, np.array(season_stats[optimise_on])))

    # Define the objective function
    fpl_problem += lpSum([player_points[i] * x[i] for i in players])

    # If the player count is more than maximum formation, use the absolute maximum formation for a team.
    if player_count > sum(max_formation):
        min_formation = __MAX_TEAM_FORMATION
        max_formation = __MAX_TEAM_FORMATION

    # Constraints
    constraints = dict(zip([p +' Min' for p in __POS_NAMES] + [p + ' Max' for p in __POS_NAMES],
                           min_formation + max_formation))

    constraints['Team Count'] = 3
    constraints['Include Count'] = len(include)

    # Prepare dictionaries for linear equation
    player_cost = dict(zip(season_stats.index, season_stats['Current Cost']))
    player_gk = dict(zip(season_stats.index, season_stats['_GK?']))
    player_def = dict(zip(season_stats.index, season_stats['_DEF?']))
    player_mid = dict(zip(season_stats.index, season_stats['_MID?']))
    player_fwd = dict(zip(season_stats.index, season_stats['_FWD?']))
    player_include = dict(zip(season_stats.index, season_stats['_Include?']))
    player_exclude = dict(zip(season_stats.index, season_stats['_Exclude?']))

    # Apply the constraints
    if budget is not None:
        fpl_problem += lpSum([player_cost[i] * x[i] for i in players]) <= budget

    gk_sum = [player_gk[i] * x[i] for i in players]
    def_sum = [player_def[i] * x[i] for i in players]
    mid_sum = [player_mid[i] * x[i] for i in players]
    fwd_sum = [player_fwd[i] * x[i] for i in players]

    fpl_problem += lpSum(gk_sum) >= constraints['GK Min']
    fpl_problem += lpSum(gk_sum) <= constraints['GK Max']
    fpl_problem += lpSum(def_sum) >= constraints['DEF Min']
    fpl_problem += lpSum(def_sum) <= constraints['DEF Max']
    fpl_problem += lpSum(mid_sum) >= constraints['MID Min']
    fpl_problem += lpSum(mid_sum) <= constraints['MID Max']
    fpl_problem += lpSum(fwd_sum) >= constraints['FWD Min']
    fpl_problem += lpSum(fwd_sum) <= constraints['FWD Max']
    fpl_problem += lpSum([player_include[i] * x[i] for i in players]) == constraints['Include Count']
    fpl_problem += lpSum([player_exclude[i] * x[i] for i in players]) == 0

    if recommend is not None:
        player_in_team = dict(zip(season_stats.index, season_stats['_In Team?']))
        fpl_problem += lpSum([player_in_team[i] * x[i] for i in players]) == player_count - recommend
    else:
        fpl_problem += lpSum([x[i] for i in players]) == player_count

    for t in season_stats['Player Team Code']:
        player_team = dict(zip(season_stats.index, season_stats['_Team ' + str(t)]))
        fpl_problem += lpSum([player_team[i] * x[i] for i in players]) <= constraints['Team Count']

    # Solve the thing
    status = fpl_problem.solve()

    if LpStatus[status] != 'Optimal':
        raise Exception(f'An optimal solution within the budget could not be found.')

    # Filter the players data frame using the selected player ids.
    player_ids = [p for p in players if x[p].value() != 0]
    return players_df[players_df.index.isin(player_ids)]


def __get_formation(team: DF)  -> Tuple[int]:
    max_team_position_counts = (DF({'Field Position': __POS_NAMES, 'Team Player Count': __MAX_TEAM_FORMATION}).set_index('Field Position'))
    team_position_counts = team.reset_index().groupby('Field Position')[['Player Code']].count().rename(columns={'Player Code': 'Player Count'})
    return tuple(max_team_position_counts
                        .merge(team_position_counts, left_index=True, right_index=True, how='left')
                        .fillna(0)
                        ['Player Count']
                        .astype(int)
                        .values)


def __get_optimal_biased_team_recommend(players: DF, formation: Tuple[int], expens_player_count: int, budget: float,
                                        optimise_team_on: str, recommend: int, include: Tuple[str], exclude: Tuple[str]) -> DF:
    team = players[lambda df: df['In Team?'] == True]
    cheap_team_player_count = sum(formation) - expens_player_count
    cheap_team = team.head(cheap_team_player_count)
    expens_team = team.tail(-cheap_team_player_count) if cheap_team_player_count > 0 else team

    expens_team_formation = __get_formation(expens_team)
    cheap_player_names = tuple(cheap_team['Name'].values)
    cheap_player_cost = cheap_team['Current Cost'].sum()

    # Get the strongest team with budget reduced what we need to reserve for the weak part of the team.
    opt_team = (__optimise_team(players, expens_team_formation, expens_team_formation, expens_player_count, budget - cheap_player_cost, optimise_team_on, recommend, include, exclude + cheap_player_names)
                .append(cheap_team))

    # Add any removed team players again and add the recommendation column.
    return (opt_team
        .append(team[lambda df: ~df.index.isin(opt_team.index.values)])
        .assign(**{'Recommendation': lambda df: np.where(df['In Team?'] == True,
                                                    np.where(df.index.isin(opt_team.index.values), 'Keep', 'Transfer out'),
                                                    'Transfer in')}))


def __get_optimal_biased_team(players: DF, formation: Tuple[int], expens_player_count: int, budget: float,
                              optimise_team_on: str, include: Tuple[str], exclude: Tuple[str]) -> DF:
    # If we need to bias budget to a proportion of the team, first work out how much budget we need to reserve for the weak part of the team.
    cheap_player_count = sum(formation) - expens_player_count
    cheap_player_cost = (players.groupby('Field Position')['Current Cost'].min().max()) * cheap_player_count

    # Get the strongest team with budget reduced what we need to reserve for the weak part of the team.
    opt_team = __optimise_team(players, __MIN_SEL_FORMATION, __MAX_SEL_FORMATION, expens_player_count, budget - cheap_player_cost, optimise_team_on, None, include, exclude)

    if cheap_player_cost > 0:
        # Work out what the formation for the weak part of the team is based on the formation of the strong part of the team.
        opt_team_formation = __get_formation(opt_team)
        cheap_team_formation = tuple(np.subtract(formation, opt_team_formation))

        # Get the best team we can get with the budget and formation of the weak team.
        cheap_team = __optimise_team(players[~players.index.isin(opt_team.index.values)], cheap_team_formation, cheap_team_formation,
                                     sum(formation) - expens_player_count, cheap_player_cost, optimise_team_on, None, include, exclude)

        # Combined both parts of the team into one.
        opt_team = pd.concat([opt_team, cheap_team])

    return opt_team


def get_new_team(players: DF) -> DF:
    """
    Gets only the members of the new team from a set of players based on the ``Recommendation`` column, i.e. players whose
    ``Recommendation`` values is either ``Transfer in`` or ``Keep``. If the column is not present in ``players``, the given data frame is returned as is.

    Args:
        players: Data frame with onr row per player.

    Returns: ``players`` data frame filtered by the new team members.
    """
    if 'Recommendation' in players.columns:
        players = players[lambda df: df['Recommendation'].isin(['Transfer in', 'Keep'])]

    return players


def get_old_team(players: DF) -> DF:
    """
    Gets only the members of the old team from a set of players based on the ``Recommendation`` column, i.e. players whose
    ``Recommendation`` values is either ``Transfer out`` or ``Keep``. If the column is not present in ``players``, the given data frame is returned as is.

    Args:
        players: Data frame with onr row per player.

    Returns: ``players`` data frame filtered by the old team members.
    """
    if 'Recommendation' in players.columns:
        players = players[lambda df: df['Recommendation'].isin(['Transfer out', 'Keep'])]

    return players


def get_optimal_team(players: DF, formation: Tuple[int] = (2, 5, 5, 3), expens_player_count: int = None, budget: float = 100.0,
                     optimise_team_on: str = 'Total Points', optimise_sel_on: str = None, recommend: int = None,
                     include: Tuple[str] = (), exclude: Tuple[str] = (), risk: float = 0.2) -> DF:
    """
    For the given formation and player data frame tries to maximise the total in the given column so that
    the current cost is with the given budget. It also adds the ``Captain/Vice Captain`` column to indicate whether a specific player
    should be captain or vice captain.

    Args:
        players: Player data frame with at least the following columns: ``Current Cost``, ``News And Date``, ``Field Position``, ``Player Team ID`` and the column specified in ``optimise_on``.
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
        The players from ``players`` with the highest value in the ``optimise_on`` column for the given formation that is within the budget.
    """

    def valid_column_not_nan(df, df_name, col_name):
        if col_name in df.columns and df[col_name].isnull().values.any():
            raise ValueError(f'At least on of the entries in the {col_name} column of the {df_name} data frame is NaN. The {col_name} must be set for all players.')

    def valid_include_exclude(df, include_exclude, include_exclude_text):
        include_exclude_lower = [v.lower() for v in include_exclude]
        if not set(df['Name'].str.lower().values) >= set(include_exclude_lower):
            raise ValueError(f'{list(set(include_exclude_lower) - set(df["Name"].str.lower().values))} in {include_exclude_text} is not in the Name column of players.')

    def risk_adj_col(df, col, risk):
        return df.assign(**{col + ' Risk Adj': lambda df: df[col] * (df['Stats Completeness Percent'] / 100 * (1 - risk) + risk)})

    required_columns = (['Current Cost', 'News And Date', 'Field Position', 'Player Team Code', 'Stats Completeness Percent', optimise_team_on, optimise_sel_on]
                        + (['In Team?'] if recommend is not None else [])
                        + ([optimise_sel_on] if optimise_sel_on is not None else []))

    if risk < 0 or risk > 1:
        raise ValueError(f'The risk argument has to be between 0 and 1 but is {risk}.')

    if not set(players.columns) >= set(required_columns):
        raise ValueError(
            f'players must at least include the following columns: {required_columns},  {list(set(required_columns) - set(players.columns))} are missing. Please ensure the data frame contains these columns.')

    valid_include_exclude(players, exclude, 'exclude')
    valid_include_exclude(players, include, 'include')

    valid_column_not_nan(players, 'players', 'Current Cost')
    valid_column_not_nan(players, 'players', optimise_team_on)
    valid_column_not_nan(players, 'players', optimise_sel_on)
    valid_column_not_nan(players, 'players', 'Field Position')
    valid_column_not_nan(players, 'players', 'Name')
    valid_column_not_nan(players, 'players', 'Stats Completeness Percent')

    if reduce(lambda x, y: x or y, map(lambda v: formation[v] > __MAX_TEAM_FORMATION[v], range(0,3))):
        raise ValueError(f'The number of players in the formation {formation} is not a subset of the maximum formation {__MAX_TEAM_FORMATION}.')

    player_count = sum(formation)
    if expens_player_count is not None and player_count < expens_player_count:
        raise ValueError(f'The number of players in the formation {formation} must be greater or equal to the number of the players in the expensive team {expens_player_count}.')

    if expens_player_count is not None and expens_player_count < sum(__MIN_SEL_FORMATION):
        raise ValueError(f'The number of players in the expensive team {expens_player_count} must be greater or equal the players in the minimum formation {sum(__MIN_SEL_FORMATION)}.')

    # Calculate risk adjusted columns
    players = (players
               .pipe(risk_adj_col, optimise_team_on, risk)
               .pipe(risk_adj_col, optimise_sel_on, risk))

    if expens_player_count is None:
        expens_player_count = sum(formation)

    if recommend is not None:
        optimised_team = __get_optimal_biased_team_recommend(players, formation, expens_player_count, budget, optimise_team_on + ' Risk Adj', recommend, include, exclude)
    else:
        optimised_team = __get_optimal_biased_team(players, formation, expens_player_count, budget, optimise_team_on + ' Risk Adj', include, exclude)

    optimised_new_team = optimised_team.pipe(get_new_team)

    # Check whether a valid solution could be found
    #if optimised_new_team['Current Cost'].sum() > budget + 0.01:
    #    raise Exception(f'An optimal solution within the budget could not be found.')

    if optimise_sel_on is not None:
        # Then select the best players from the team to play, i.e. the best valid formation
        selected_team = __optimise_team(optimised_new_team, __MIN_SEL_FORMATION, __MAX_SEL_FORMATION, 11, None, optimise_sel_on + ' Risk Adj')

        optimised_team = (optimised_team
                            .assign(**{'Selected?': lambda df: optimised_team.index.map(lambda x: x in selected_team.index.values)})
                            .sort_values(['Selected?', optimise_sel_on + ' Risk Adj'], ascending=False)
                            .assign(**{'Captain?': False})
                            .assign(**{'Vice Captain?': False})
                          )

        optimised_team.iloc[0, optimised_team.columns.get_loc('Captain?')] = True
        if optimised_team.shape[0] > 1:
            optimised_team.iloc[1, optimised_team.columns.get_loc('Vice Captain?')] = True

        optimised_team = (optimised_team
                          .assign(**{'Point Factor': lambda df: df.apply(lambda row: 0 if not row['Selected?'] else 1 if not row['Captain?'] else 2, axis=1)})
                          .sort_values(['Field Position', optimise_sel_on + ' Risk Adj']))

    return optimised_team.pipe(__remove_temp_cols)
