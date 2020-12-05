import collections
import requests, io
from pandas.api.types import is_string_dtype
from common import *
from typing import List, Dict, Union

# Define type aliases
DF = pd.DataFrame
S = pd.Series


def get_next_gw_counts(ctx: Context) -> Dict[str, int]:
    return collections.OrderedDict([(get_next_gw_name(gw, ctx), gw) for gw in range(1, ctx.total_gws - ctx.next_gw + 1)])


def get_next_gw_name(next_gw: int, ctx: Context) -> str:
    if next_gw == 1:
        return 'Next GW'

    return f'Next {next_gw} GWs'


def get_team_fixtures_by_gw(team_fixture_strength: DF, value_col: str, ctx: Context) -> DF:
    team_fixture_strength = (team_fixture_strength.assign(**{'Label':
                                         lambda df: np.where(df['Is Home?'], 'Home: ', 'Away: ')
                                                    + df['Team Short Name'] + ' (' + df['Team FDR'].astype(str) + ') -> '
                                                    + df['Opp Team Short Name'] + ' (' + df['Opp Team FDR'].astype(str) + ')'}))

    agg_func = np.mean
    if (is_string_dtype(team_fixture_strength[value_col])):
        agg_func = lambda v: ', '.join(v)

    team_fixtures_by_gw = (team_fixture_strength
                           [lambda df: (df['Season'] == ctx.current_season) & (df['Game Week'] >= 1)]
                           .rename({'Game Week': 'GW'})
                           .pivot_table(index='Team Short Name', columns=['Season', 'Game Week'], values=value_col, aggfunc=agg_func)
                           .sort_index(ascending=False))

    team_fixtures_by_gw.columns = [' '.join([col[0], f', GW{col[1]}']).strip() for col in team_fixtures_by_gw.columns]
    return team_fixtures_by_gw


def get_players_id_code_map(players: DF) -> S:
    return players[['Player ID']].reset_index().set_index('Player ID')['Player Code']


def is_date_col(col: str) -> bool:
    return 'Last Updated' in col


def get_df(url: str, index: Union[str, List[str]] = None) -> DF:
    res = requests.get(url)
    df = pd.read_csv(io.BytesIO(res.content), low_memory=False)

    for col in filter(is_date_col, df):
        df = df.assign(**{col: lambda df: pd.to_datetime(df[col])})

    if index is not None:
        df = df.set_index(index, drop=True)

    return df