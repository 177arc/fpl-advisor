"""
This module contains functions for back testing the expected points predictions.
"""

from common import *
from optimiser import *

# Define type aliases
DF = pd.DataFrame
S = pd.Series


def filter_gw(player_fixture_stats: DF, gw: int, ctx: Context) -> DF:
    return (player_fixture_stats
            [lambda df: (df['Game Week'] == gw) & (df['Season'] == ctx.current_season)]
            .assign(**{'Current Cost': lambda df: np.where(~df['Fixture Cost'].isnull(), df['Fixture Cost'], df['Current Cost'])})
            .reset_index()
            .set_index('Player Code'))


def prep_gw(player_gw_eps_gw: DF, player_teams: DF, ep_column: str = 'Expected Points') -> DF:
    player_team_eps_gw = (player_gw_eps_gw
                          [['Fixture Total Points', ep_column, 'Current Cost', 'Stats Completeness Percent']]
                          [lambda df: ~df[ep_column].isnull()]
                          .merge(player_teams[['Name', 'Field Position Code', 'Field Position', 'Player Team Code', 'First Name', 'Last Name', 'ICT Index', 'Team Short Name', 'Name and Short Team', 'Minutes Percent']],
                                 left_index=True, right_index=True, suffixes=(False, False))
                          )
    player_team_eps_gw['News And Date'] = None  # Unfortunately, we don't have historic news information.
    return player_team_eps_gw[((player_team_eps_gw['Fixture Total Points'].isnull()) | (player_team_eps_gw['Fixture Total Points'] > 0))
                              & (player_team_eps_gw['Minutes Percent'] > 50)]


def get_optimal_team_exp(player_team_exp_gw: DF, ep_column: str = 'Expected Points',
                         formation: str = '2-5-5-3', budget: float = 100.0) -> DF:
    player_team_optimal = get_optimal_squad(player_team_exp_gw,
                                            optimise_team_on=ep_column,
                                            optimise_sel_on=ep_column,
                                            formation=formation,
                                            budget=budget)
    return player_team_optimal[['Fixture Total Points', ep_column, 'Point Factor', 'Selected?']]


def get_optimal_team_act(players_history_fixtures_gw: DF,
                         formation: str = '2-5-5-3', budget: float = 100.0) -> DF:
    player_team_optimal_act = get_optimal_squad(players_history_fixtures_gw,
                                                optimise_team_on='Fixture Total Points',
                                                optimise_sel_on='Fixture Total Points',
                                                formation=formation,
                                                budget=budget)
    return player_team_optimal_act[['Fixture Total Points', 'Selected?']]


def calc_team_points(player_team: DF, points_col: str = 'Fixture Total Points', sel_only: bool = True) -> float:
    player_team = player_team.copy()

    if sel_only:
        player_team = player_team[player_team['Selected?']]

    player_team['Points'] = player_team[points_col]

    if 'Point Factor' in player_team.columns.values:
        player_team['Points'] *= player_team['Point Factor']

    return player_team['Points'].sum()


def calc_stats(player_team_optimal_exp: DF, pred_kind: str, ep_column: str) -> dict:
    mae = abs(player_team_optimal_exp['Fixture Total Points'] - player_team_optimal_exp[ep_column]).mean()
    mse = ((player_team_optimal_exp['Fixture Total Points'] - player_team_optimal_exp[ep_column]) ** 2).mean()
    return {f'{pred_kind} Expected Points': calc_team_points(player_team_optimal_exp, ep_column),
            f'{pred_kind} Actual Points': calc_team_points(player_team_optimal_exp, 'Fixture Total Points'),
            f'{pred_kind} Mean Absolute Error': mae, f'{pred_kind} Mean Square Error': mse}


def pred_free_hit_gw(players_gw_team_eps: DF, player_teams: DF, team_budget: float, gw: int, ctx: Context) -> S:
    # noinspection PyTypeChecker
    eps = (players_gw_team_eps
           .reset_index()
           .pipe(filter_gw, gw, ctx)
           .pipe(prep_gw, player_teams, 'Expected Points')
           .pipe(get_optimal_team_exp, 'Expected Points', '2-5-5-3', team_budget)
           .pipe(calc_team_points, 'Expected Points'))
    return S([gw, eps], index=['Game Week', 'Expected Points'])

def pred_bench_boost_gw(player_team_eps_user: DF, player_teams: DF, team_budget, gw: int, ctx: Context) -> S:
    # noinspection PyTypeChecker
    eps = (player_team_eps_user
           .reset_index()
           .pipe(filter_gw, gw, ctx)
           .pipe(prep_gw, player_teams, 'Expected Points')
           .pipe(get_optimal_team_exp, 'Expected Points', '2-5-5-3', team_budget)
           .pipe(calc_team_points, points_col='Expected Points', sel_only=False))
    return S([gw, eps], index=['Game Week', 'Expected Points'])

def back_test_gw(players_gw_team_eps: DF, gw: int, player_teams: DF, ctx: Context) -> dict:
    # noinspection PyTypeChecker
    player_gw_eps_gw = (players_gw_team_eps
                        .pipe(filter_gw, gw, ctx)
                        [lambda df: ~df['Fixture Total Points'].isnull()])

    player_team_optimal_act = (player_gw_eps_gw
                               .pipe(get_optimal_team_act))

    calc_nn_prep_gw = (player_gw_eps_gw
                       .pipe(prep_gw, player_teams, 'Expected Points'))

    results = {'Game Week': gw, 'Actual Points Dream Team': player_team_optimal_act.pipe(calc_team_points)}
    results = {**results, **calc_nn_prep_gw.pipe(get_optimal_team_exp, 'Expected Points').pipe(calc_stats, 'Calc', 'Expected Points')}

    return results


def add_gws_ago(players_gw_team: DF) -> DF:
    return (players_gw_team
            .sort_values(['Season', 'Game Week'], ascending=False)
            .assign(**{'GWs Ago': lambda df: (~df['Expected Points'].isnull()).cumsum()}))


def get_gw_points_backtest(players_gw_team_eps: DF, ctx: Context) -> DF:
    return (players_gw_team_eps
         .reset_index()
         [lambda df: (df['Fixture Minutes Played'] > 0) & (df['Fixtures Played To Fixture'] > 4) &  # (df['Total Points To Fixture'] > 48) &
              ((df['Season'] == ctx.current_season) & (df['Game Week'] < ctx.next_gw) | (df['Season'] != ctx.current_season))]
        .assign(**{'Player Fixture Error': lambda df: np.abs(df['Expected Points']-df['Fixture Total Points'])})
        .assign(**{'Player Fixture Error Simple': lambda df: np.abs(df['Expected Points Simple']-df['Fixture Total Points'])})
        .assign(**{'Player Fixture Sq Error': lambda df: df['Player Fixture Error']**2})
        .assign(**{'Player Fixture Sq Error Simple': lambda df: df['Player Fixture Error Simple']**2})
        .groupby(['Season', 'Game Week'])
         [['Player Fixture Error', 'Player Fixture Sq Error', 'Expected Points', 'Player Fixture Error Simple', 'Player Fixture Sq Error Simple', 'Expected Points Simple', 'Fixture Total Points', 'Player Code']]
         .agg({'Player Fixture Error': 'sum', 'Player Fixture Sq Error': 'sum', 'Expected Points': 'sum', 'Player Fixture Error Simple': 'sum', 'Player Fixture Sq Error Simple': 'sum', 'Expected Points Simple': 'sum', 'Fixture Total Points': 'sum', 'Player Code': 'count'})
         .reset_index()
         .pipe(add_gws_ago)
         [lambda df: df['GWs Ago'] <= ctx.fixtures_look_back]
         .sort_values(['Season', 'Game Week'])
         .rename(columns={'Player Code': 'Player Count'})
         .assign(**{'Avg Expected Points': lambda df: df['Expected Points']/df['Player Count']})
         .assign(**{'Avg Expected Points Simple': lambda df: df['Expected Points Simple']/df['Player Count']})
         .assign(**{'Avg Fixture Total Points': lambda df: df['Fixture Total Points']/df['Player Count']})
         .assign(**{'Error': lambda df: df['Player Fixture Error']/df['Player Count']})
         .assign(**{'Error Simple': lambda df: df['Player Fixture Error Simple']/df['Player Count']})
         .assign(**{'Sq Error': lambda df: df['Player Fixture Sq Error']/df['Player Count']})
         .assign(**{'Sq Error Simple': lambda df: df['Player Fixture Sq Error Simple']/df['Player Count']})
         .assign(**{'Season Game Week': lambda df: df['Season']+', GW '+df['Game Week'].apply('{:.0f}'.format)}))