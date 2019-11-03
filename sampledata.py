import logging as log
import pandas as pd
from fplpandas import FPLPandas
import os

log.basicConfig(level=log.INFO, format='%(message)s')

def write_df(df: pd.DataFrame, file_path: str) -> None:
    df.to_csv(file_path)
    log.info(f'Written sample data to {file_path}.')

def get_data_samples() -> None:
    """
    Extracts sample data from the API and writes it to the samples directory so the available data sets are easier to understand.
    """
    os.makedirs('samples', exist_ok=True)

    fpl = FPLPandas(6769616, 'fpl@177arc.net', 'TestMcTestFace')
    write_df(fpl.get_teams(), 'samples/get_teams.csv')
    write_df(fpl.get_fixtures(), 'samples/get_fixtures.csv')
    write_df(fpl.get_players(), 'samples/get_players.csv')
    write_df(fpl.get_player(1) ,'samples/get_player_with_id_1.csv')
    write_df(fpl.get_player_fixtures(1), 'samples/get_player_fixtures_with_id_1.csv')
    write_df(fpl.get_player_history(1), 'samples/get_player_history_with_id_1.csv')
    write_df(fpl.get_player_past_seasons(1), 'samples/get_player_past_seasons_with_id_1.csv')
    write_df(fpl.get_user_team(), 'samples/get_user_team.csv')

if __name__ == '__main__':
    get_data_samples()
