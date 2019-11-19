import pandas as pd
import warnings

next_gws = ['Next GW', 'Next 5 GWs', 'GWs To Half', 'GWs To End']
position_by_type = {1: 'GK',  2: 'DEF',  3: 'MID',  4:  'FWD'}

def summarise_team(team: pd.DataFrame) -> pd.DataFrame:
    for gw in next_gws:
        team['Expected Points ' + gw] = team.apply(lambda row: row['Expected Points ' + gw] * max(row['Point Factor'], 1), axis=1)

    aggr = {'Current Cost': 'sum'}
    aggr = {**aggr, **{'Expected Points ' + gw: 'sum' for gw in next_gws}}
    aggr = {**aggr, **{'Total Points Consistency': 'mean'}}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        team_summary = team.agg(aggr).to_frame().T
        team_summary['Team'] = 'All Players'
        team_sel_summary = team[team['Selected?'] == True].agg(aggr).to_frame().T
        team_sel_summary['Team'] = 'Selected Players'

    summary = pd.concat([team_summary, team_sel_summary])
    return summary.set_index('Team')