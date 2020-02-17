"""
This module contains functions for deriving new FPL data set from existing ones.
"""

import pandas as pd
import datetime as dt
import re
from common import *
from datadict.jupyter import DataDict

def get_team_fixture_scores(fixture_teams: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    validate_df(fixture_teams, 'fixture_teams', ['Fixture ID', 'Game Week', 'Home Team Score', 'Away Team Score', 'Home Team ID', 'Away Team ID'])
    validate_df(teams, 'teams', ['Team Code', 'Team Short Name', 'Team Name'])

    # Unfold data frame so that there a two rows for each fixture.
    return (pd.melt(fixture_teams[['Fixture ID', 'Game Week', 'Home Team Score', 'Away Team Score', 'Home Team ID', 'Away Team ID']],
                    id_vars=['Fixture ID', 'Game Week', 'Home Team Score', 'Away Team Score'],
                    value_vars=['Home Team ID', 'Away Team ID'])
            .rename(columns={'variable': 'Variable', 'value': 'Value'})
            .sort_values(['Game Week'])
            .assign(**{'Team Goals Scored': lambda df: df.apply(lambda row: row['Home Team Score'] if row['Variable'] == 'Home Team ID' else row['Away Team Score'], axis=1)})
            .assign(**{'Team Goals Conceded': lambda df: df.apply(lambda row: row['Away Team Score'] if row['Variable'] == 'Home Team ID' else row['Home Team Score'], axis=1)})
            .assign(**{'Is Home?': lambda df: df['Variable'] == 'Home Team ID'})
            .rename(columns={'Value': 'Team ID'}).drop('Variable', axis=1)
            .merge(teams, left_on='Team ID', right_on='Team ID'))


def get_team_score_stats(team_fixture_scores: pd.DataFrame) -> pd.DataFrame:
    validate_df(team_fixture_scores, 'team_fixture_scores', ['Team ID', 'Team Short Name', 'Is Home?', 'Team Goals Scored', 'Team Goals Conceded'])

    team_score_stats = (team_fixture_scores
                        .groupby(['Team ID', 'Team Short Name', 'Is Home?'])[['Team Goals Scored', 'Team Goals Conceded']]
                        .sum()
                        .unstack(level=-1)
                        .reset_index()
                        .rename(columns={False: 'Away', True: 'Home'}))
    team_score_stats.columns = [' '.join(col).strip() for col in team_score_stats.columns.values]
    team_score_stats = (team_score_stats
                        .set_index('Team ID')
                        .rename(columns={'Team Goals Conceded Home': 'Total Team Goals Conceded Home',
                                         'Team Goals Conceded Away': 'Total Team Goals Conceded Away',
                                         'Team Goals Scored Home': 'Total Team Goals Scored Home',
                                         'Team Goals Scored Away': 'Total Team Goals Scored Away'})
                        .assign(**{'Total Team Goals Scored Ratio': lambda df: df['Total Team Goals Scored Away'] / df['Total Team Goals Scored Home']})
                        .assign(**{'Total Team Goals Conceded Ratio': lambda df: df['Total Team Goals Conceded Away'] / df['Total Team Goals Conceded Home']})
                        .assign(**{'Total Team Goals Scored': lambda df: df['Total Team Goals Scored Away'] + df['Total Team Goals Scored Home']})
                        .assign(**{'Total Team Goals Conceded': lambda df: df['Total Team Goals Conceded Away'] + df['Total Team Goals Conceded Home']}))

    return team_score_stats


def get_next_gw_counts(next_gws: list, next_gw: int, total_gws: int) -> dict:
    remain_gws = total_gws - next_gw + 1
    next_gw_counts = {}
    for next_gw in next_gws:
        if next_gw == 'Next GW':
            next_gw_counts[next_gw] = min(1, remain_gws)

        if next_gw == 'GWs To End':
            next_gw_counts[next_gw] = remain_gws

        gw = re.search(r'Next (\d+) GWs', next_gw, re.IGNORECASE)
        if gw:
            next_gw_counts[next_gw] = min(int(gw.group(1)), remain_gws)

    return next_gw_counts


def calc_eps_for_next_gws(df: pd.DataFrame, next_gws: list, next_gw: int, total_gws: int):
    df = df.sort_values('Game Week')
    next_gw_counts = get_next_gw_counts(next_gws, next_gw, total_gws)

    current_df = df[df['Game Week'] == next_gw]
    row = current_df.iloc[0]
    for next_gw_post_fix, next_gw_count in next_gw_counts.items():
        future_df = df[(df['Game Week'] >= next_gw) & (df['Game Week'] < next_gw + next_gw_count)]
        row['Expected Points ' + next_gw_post_fix] = (future_df['Expected Points'] * future_df['Chance Avail Next GW'] / 100).sum()
        if next_gw_post_fix != 'GWs To End':
            row['Fixtures ' + next_gw_post_fix] = value_or_default(future_df['Fixture Short Name Difficulty'].str.cat(sep=', '))

    return row


def calc_eps(player_fixture_stats: pd.DataFrame) -> pd.DataFrame:
    return ((player_fixture_stats['Total Points To GW']/player_fixture_stats['GWs Played To GW']).fillna(player_fixture_stats['Total Points To GW'])
               *player_fixture_stats['Rel. Fixture Strength']/player_fixture_stats['Rel. Fixture Strength To GW'].fillna(method='ffill'))


def calc_player_fixture_stats(players_fixture_team_points: pd.DataFrame):
    return (players_fixture_team_points
        .sort_values('Kick Off Time')
        .assign(**{'GW Played': lambda df: df['Game Minutes Played'] > 0})
        .assign(**{'GWs Played To GW': lambda df: df.groupby('Player ID')['GW Played'].apply(lambda x: x.shift().cumsum().fillna(method='ffill'))})
        .assign(**{'Total Points To GW': lambda df: df.groupby('Player ID')['Game Total Points'].apply(lambda x: x.shift().cumsum().fillna(method='ffill'))})
        .assign(**{'Total Opp Team Goals Scored Diff': lambda df: df['Total Team Goals Scored']-df['Total Opp Team Goals Scored']})
        .assign(**{'Avg Points To GW': lambda df: df.groupby('Player ID')['Game Total Points'].apply(lambda x: x.shift().rolling(10, min_periods=1).mean().fillna(method='ffill'))})
        .assign(**{'Avg Minutes Played Recently To GW': lambda df: df.groupby('Player ID')['Game Minutes Played'].apply(lambda x: x.shift().rolling(10, min_periods=1).mean())})
        .assign(**{'Avg Points Opp Points Adj To GW': lambda df: df.apply(lambda row: row['Avg Points To GW']*row['Team Total Points']/row['Opp Team Total Points'], axis=1)}))


def get_player_teams(players: pd.DataFrame, teams: pd.DataFrame, dd: DataDict):
    return (players
    .reset_index(drop=False)
    .merge(teams, left_on='Player Team ID', right_on='Team ID')
    .set_index('Player ID')
    .assign(**{'Long Name': lambda df: df['First Name']+' '+df['Last Name']})
    .assign(**{'Long Name and Team': lambda df: df['Long Name']+' ('+df['Team Name']+')'})
    .assign(**{'Name and Short Team': lambda df: df['Name']+' ('+df['Team Short Name']+')'})
    .pipe(dd.reorder))

def get_fixture_teams(fixtures: pd.DataFrame, teams: pd.DataFrame, dd: DataDict) -> pd.DataFrame:
    return (fixtures
         .reset_index()
         .merge(teams[['Team Name', 'Team Short Name']]
                    .rename(columns={'Team Name': 'Team Name Home', 'Team Short Name': 'Team Short Name Home'}),
                left_on='Home Team ID',
                right_on='Team ID', suffixes=(False, False))
         .merge(teams[['Team Name', 'Team Short Name']]
                    .rename(columns={'Team Name': 'Team Name Away', 'Team Short Name': 'Team Short Name Away'}),
                left_on='Away Team ID',
                right_on='Team ID', suffixes=(False, False))
         .pipe(dd.reorder))

def get_news(row: pd.Series) -> str:
    '''Derives the text for the News column.'''
    if pd.isnull(row['news']) or row['news'] == '': return None
    date_part = '' if pd.isnull(row['news_added'] or row['news_added'] == 'None') else ' ('+dt.datetime.strftime(dt.datetime.strptime(re.sub(r'\.\d+', '', row['news_added']), '%Y-%m-%dT%H:%M:%SZ'), '%d %b %Y')+')'
    return str(row['news'])+date_part


def prepare_players(players_raw: pd.DataFrame, dd: DataDict) -> pd.DataFrame:
    return (players_raw
        .pipe(dd.remap, data_set='player')
        .assign(**{'ICT Index': lambda df: pd.to_numeric(df['ICT Index'])})
        .assign(**{'Field Position': lambda df: df['Field Position ID'].map(lambda x: position_by_type[x])})
        .assign(**{'Current Cost': lambda df: df['now_cost']/10})
        .assign(**{'Minutes Percent': lambda df: df['Minutes Played']/df['Minutes Played'].max()*100})
        .assign(**{'News And Date': lambda df: df.apply(lambda row: get_news(row), axis=1)})
        .assign(**{'Percent Selected': lambda df: pd.to_numeric(df['Percent Selected'])})
        .assign(**{'Chance Avail This GW': lambda df: df['Chance Avail This GW'].map(lambda x: x if not pd.isnull(x) else 100)})
        .assign(**{'Chance Avail Next GW': lambda df: df['Chance Avail Next GW'].map(lambda x: x if not pd.isnull(x) else 100)})
        .rename_axis('Player ID')
        .pipe(dd.reorder))

def prepare_players_history_past(players_history_past_raw: pd.DataFrame, dd: DataDict) -> pd.DataFrame:
    return (players_history_past_raw
         .pipe(dd.remap, data_set='players_history_past')
         .rename_axis(['Player ID', 'Season']))

def prepare_players_history(players_history_raw: pd.DataFrame, dd: DataDict) -> pd.DataFrame:
    return (players_history_raw
        .pipe(dd.remap, 'player_hist')
        .rename_axis(['Player ID', 'Fixture ID'])
        .assign(**{'Game Cost': lambda df: df['value']/10})
        .assign(**{'Game ICT Index': lambda df: pd.to_numeric(df['Game ICT Index'])}))

def prepare_teams(teams_raw: pd.DataFrame, dd: DataDict) -> pd.DataFrame:
    return (teams_raw
    .pipe(dd.remap, 'team')
    .rename_axis('Team ID')
    .drop(columns=['Strength Attack Home', 'Strength Attack Away', 'Strength Defence Home', 'Strength Defence Away', 'Strength Overall Home', 'Strength Overall Away'], errors='ignore'))

def prepare_fixtures(fixtures_raw: pd.DataFrame, dd: DataDict) -> pd.DataFrame:
    return (fixtures_raw
        .drop(columns=['stats', 'finished_provisional', 'provisional_start_time'])
        .pipe(dd.remap, 'fixture')
        .rename_axis('Fixture ID')
        .pipe(dd.reorder))