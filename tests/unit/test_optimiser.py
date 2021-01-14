import unittest
import pandas as pd, numpy as np
import optimiser
import warnings
import os


class TestOptimiser(unittest.TestCase):
    """
    Unit tests the optimiser functions.
    """

    test_file = os.path.join(os.path.dirname(__file__), 'test_data_optimiser.csv')

    def __assert_team(self, team, budget):
        self.assertEqual(team.shape[0], 15, msg='Row count must be 15.')

        position_counts = team[['Field Position', 'Name']].groupby('Field Position').count()['Name'].to_dict()
        self.assertDictEqual(position_counts, {'DEF': 5, 'FWD': 3, 'GK': 2, 'MID': 5}, msg='Team composition incorrect.')

        self.assertLessEqual(team['Current Cost'].sum()-0.01, budget, msg='Budget breached.')
        self.assertGreaterEqual(team['Current Cost'].sum(), budget*0.95, msg='Budget not used well enough.')

        self.assertEqual(team[team['Captain?'] == True]['Captain?'].count(), 1, msg='More than one Captain')
        self.assertEqual(team[team['Vice Captain?'] == True]['Vice Captain?'].count(), 1, msg='More than one Vice Captain')
        self.assertEqual(team['Selected?'].isin([True]).sum(), 11, msg='Not 11 players selected.')

        self.assertTrue(team[(team['Selected?'] == True) & (team['Captain?'] == False)]['Point Factor'].eq(1).all())
        self.assertTrue(team[(team['Selected?'] == True) & (team['Captain?'] == True)]['Point Factor'].eq(2).all())
        self.assertTrue(team[team['Selected?'] == False]['Point Factor'].eq(0).all())

    def test_optimal_team_high_risk(self):
        players = pd.read_csv(self.test_file).set_index('Player Code')

        budget=100.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            team = optimiser.get_optimal_team(players, (2, 5, 5, 3), budget=budget, optimise_team_on='Expected Points Next 5 GWs', optimise_sel_on='Expected Points Next GW', risk=1)

        self.__assert_team(team, budget)

        self.assertListEqual(list(team['Name'].values), ['Kelly', 'Alderweireld', 'Matip', 'Ward', 'Alexander-Arnold', 'Ayew', 'Abraham', 'Firmino', 'Lloris', 'Ederson', 'Kovacic', 'Son', 'Mount', 'David Silva', 'De Bruyne'])
        self.assertTrue(team[team['Name'] == 'De Bruyne']['Captain?'].iloc[0])
        self.assertTrue(team[team['Name'] == 'Abraham']['Vice Captain?'].iloc[0])

        self.assertListEqual(list(team[team['Selected?'] == True]['Name'].values),  ['Matip', 'Ward', 'Alexander-Arnold', 'Ayew', 'Abraham', 'Firmino', 'Ederson', 'Son', 'Mount', 'David Silva', 'De Bruyne'])

    def test_optimal_team_low_risk(self):
        players = pd.read_csv(self.test_file).set_index('Player Code')

        budget=100.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            team = optimiser.get_optimal_team(players, (2, 5, 5, 3), budget=budget, optimise_team_on='Expected Points Next 5 GWs', optimise_sel_on='Expected Points Next GW', risk=0)

        self.__assert_team(team, budget)

        self.assertListEqual(list(team['Name'].values), ['Kelly', 'Alderweireld', 'Ward', 'van Aanholt', 'Alexander-Arnold', 'Origi', 'Abraham', 'Firmino', 'Lloris', 'Ederson', 'Kovacic', 'Son', 'Mount', 'David Silva', 'De Bruyne'])
        self.assertTrue(team[team['Name'] == 'De Bruyne']['Captain?'].iloc[0])
        self.assertTrue(team[team['Name'] == 'Abraham']['Vice Captain?'].iloc[0])

        self.assertListEqual(list(team[team['Selected?'] == True]['Name'].values),  ['Alderweireld', 'Ward', 'van Aanholt', 'Alexander-Arnold', 'Abraham', 'Firmino', 'Ederson', 'Son', 'Mount', 'David Silva', 'De Bruyne'])

    def test_optimal_team_include(self):
        players = pd.read_csv(self.test_file).set_index('Player Code')

        budget=100.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            team = optimiser.get_optimal_team(players, (2, 5, 5, 3), budget=budget, optimise_team_on='Expected Points Next 5 GWs', optimise_sel_on='Expected Points Next GW', include=['Mané'], risk=1)

        self.__assert_team(team, budget)

        self.assertListEqual(list(team['Name'].values), ['Rose', 'Kelly', 'Alderweireld', 'Matip', 'Ward', 'Ayew', 'Abraham', 'Firmino', 'Lloris', 'Ederson', 'Kovacic', 'Mount', 'David Silva', 'Mané', 'De Bruyne'])

    def test_optimal_team_include_invalid_name(self):
        players = pd.read_csv(self.test_file).set_index('Player Code')

        with self.assertRaisesRegex(ValueError, r'\[\'Marc Maier\'\] in include is not in the Name column of players.'):
            optimiser.get_optimal_team(players, (2, 5, 5, 3), budget=100.0, optimise_team_on='Expected Points Next 5 GWs', optimise_sel_on='Expected Points Next GW', include=['Marc Maier'])

    def test_optimal_team_exclude(self):
        players = pd.read_csv(self.test_file).set_index('Player Code')

        budget=100.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            team = optimiser.get_optimal_team(players, (2, 5, 5, 3), budget=budget, optimise_team_on='Expected Points Next 5 GWs', optimise_sel_on='Expected Points Next GW', exclude=['Ederson'], risk=1)

        self.__assert_team(team, budget)
        self.assertListEqual(list(team['Name'].values), ['Alderweireld', 'van Dijk', 'Ward', 'van Aanholt', 'Alexander-Arnold', 'Kane', 'Ayew', 'Abraham', 'Adrián', 'Lloris', 'Kovacic', 'Mount', 'Bernardo Silva', 'David Silva', 'De Bruyne'])

    def test_optimal_team_include_exclude(self):
        players = pd.read_csv(self.test_file).set_index('Player Code')

        budget=100.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            team = optimiser.get_optimal_team(players, (2, 5, 5, 3), budget=budget, optimise_team_on='Expected Points Next 5 GWs', optimise_sel_on='Expected Points Next GW', include=['Mané'], exclude=['Ederson'], risk=1)

        self.__assert_team(team, budget)
        self.assertListEqual(list(team['Name'].values), ['Kelly', 'Alderweireld', 'Walker', 'Ward', 'Alexander-Arnold', 'Ayew', 'Abraham', 'Firmino', 'Whiteman', 'Lloris', 'Kovacic', 'Mount', 'David Silva', 'Mané', 'De Bruyne'])

    def test_optimal_team_include_exclude_different_capitalisation(self):
        players = pd.read_csv(self.test_file).set_index('Player Code')

        budget=100.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            team = optimiser.get_optimal_team(players, (2, 5, 5, 3), budget=budget, optimise_team_on='Expected Points Next 5 GWs', optimise_sel_on='Expected Points Next GW', include=['mAné'], exclude=['EDErson'], risk=1)

        self.__assert_team(team, budget)
        self.assertListEqual(list(team['Name'].values), ['Kelly', 'Alderweireld', 'Walker', 'Ward', 'Alexander-Arnold', 'Ayew', 'Abraham', 'Firmino', 'Whiteman', 'Lloris', 'Kovacic', 'Mount', 'David Silva', 'Mané', 'De Bruyne'])

    def test_optimal_team_expens_player_count(self):
        players = pd.read_csv(self.test_file).set_index('Player Code')

        budget=100.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            team = optimiser.get_optimal_team(players, formation=(2, 5, 5, 3), expens_player_count=11, budget=budget, optimise_team_on='Expected Points Next 5 GWs', optimise_sel_on='Expected Points Next GW', risk=1)

        self.__assert_team(team, budget)

        self.assertListEqual(list(team['Name'].values), ['Walker-Peters', 'Kelly', 'Ward', 'van Aanholt', 'Alexander-Arnold', 'Abraham', 'Firmino', 'Agüero', 'Whiteman', 'Lloris', 'McArthur', 'Wijnaldum', 'Mount', 'David Silva', 'De Bruyne'])
        self.assertTrue(team[team['Name'] == 'Agüero']['Captain?'].iloc[0])
        self.assertTrue(team[team['Name'] == 'De Bruyne']['Vice Captain?'].iloc[0])

    def test_optimal_team_expens_player_count_recommend(self):
        players = pd.read_csv(self.test_file).set_index('Player Code')

        budget=100.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            team = optimiser.get_optimal_team(players, formation=(2, 5, 5, 3), expens_player_count=11, budget=budget, optimise_team_on='Expected Points Next 5 GWs', optimise_sel_on='Expected Points Next GW', risk=1, recommend=1)

        self.assertEqual(team.shape[0], 16, msg='Row count must be 16.')

        new_team = team[lambda df: df['Recommendation'] != 'Transfer out']
        position_counts = new_team[['Field Position', 'Name']].groupby('Field Position').count()['Name'].to_dict()
        self.assertDictEqual(position_counts, {'DEF': 5, 'FWD': 3, 'GK': 2, 'MID': 5}, msg='Team composition incorrect.')
        self.assertListEqual(list(new_team['Name'].values), ['Kelly', 'Matip', 'Ward', 'van Aanholt', 'Alexander-Arnold', 'Ayew', 'Abraham', 'Firmino', 'Lloris', 'Ederson', 'Kovacic', 'Son', 'Mount', 'David Silva', 'De Bruyne'])
        self.assertLessEqual(new_team['Current Cost'].sum(), budget)
        self.assertGreaterEqual(new_team['Current Cost'].sum(), budget*0.95)

        out_players = team[lambda df: df['Recommendation'] == 'Transfer out']
        self.assertListEqual(list(out_players['Name'].values),  ['Alderweireld'])

        in_players = team[lambda df: df['Recommendation'] == 'Transfer in']
        self.assertListEqual(list(in_players['Name'].values), ['van Aanholt'])


    def test_optimal_team_expens_player_count_too_many(self):
        players = pd.read_csv(self.test_file).set_index('Player Code')

        with self.assertRaisesRegex(ValueError, r'must be greater or equal to the number of the players in the expensive team'):
            team = optimiser.get_optimal_team(players, formation=(2, 5, 5, 3), expens_player_count=16, budget=100.0, optimise_team_on='Expected Points Next 5 GWs', optimise_sel_on='Expected Points Next GW', risk=1)


    def test_optimal_team_exclude_invalid_name(self):
        players = pd.read_csv(self.test_file).set_index('Player Code')

        with self.assertRaisesRegex(ValueError, r'\[\'Marc Maier\'\] in exclude is not in the Name column of players.'):
            optimiser.get_optimal_team(players, (2, 5, 5, 3), budget=100.0, optimise_team_on='Expected Points Next 5 GWs', optimise_sel_on='Expected Points Next GW', exclude=['Marc Maier'])


    def test_optimal_team_nan_in_cost(self):
        players = pd.read_csv(self.test_file).set_index('Player Code')
        players.at[526, 'Current Cost'] = np.nan

        with self.assertRaisesRegex(ValueError, r'At least on of the entries in the Current Cost column of the players data frame is NaN. The Current Cost must be set for all players.'):
            optimiser.get_optimal_team(players, (2, 5, 5, 3), budget=100.0, optimise_team_on='Expected Points Next 5 GWs', optimise_sel_on='Expected Points Next GW')


    def test_optimal_team_nan_in_optimise_team_on(self):
        players = pd.read_csv(self.test_file).set_index('Player Code')
        players.at[526, 'Expected Points Next 5 GWs'] = np.nan

        with self.assertRaisesRegex(ValueError, r'At least on of the entries in the Expected Points Next 5 GWs column of the players data frame is NaN. The Expected Points Next 5 GWs must be set for all players.'):
            optimiser.get_optimal_team(players, (2, 5, 5, 3), budget=100.0, optimise_team_on='Expected Points Next 5 GWs', optimise_sel_on='Expected Points Next GW')


    def test_optimal_team_nan_in_optimise_sel_on(self):
        players = pd.read_csv(self.test_file).set_index('Player Code')
        players.at[526, 'Expected Points Next GW'] = np.nan

        with self.assertRaisesRegex(ValueError, r'At least on of the entries in the Expected Points Next GW column of the players data frame is NaN. The Expected Points Next GW must be set for all players.'):
            optimiser.get_optimal_team(players, (2, 5, 5, 3), budget=100.0, optimise_team_on='Expected Points Next 5 GWs', optimise_sel_on='Expected Points Next GW')


    def test_recommend_transfer(self):
        players = pd.read_csv(self.test_file).set_index('Player Code')

        budget = 100.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            team = optimiser.get_optimal_team(players, (2, 5, 5, 3), budget=budget, optimise_team_on='Expected Points Next 5 GWs', optimise_sel_on='Expected Points Next GW', recommend=2)

        self.assertEqual(team.shape[0], 17, msg='Row count must be 17.')

        new_team = team[lambda df: df['Recommendation'] != 'Transfer out']
        position_counts = new_team[['Field Position', 'Name']].groupby('Field Position').count()['Name'].to_dict()
        self.assertDictEqual(position_counts, {'DEF': 5, 'FWD': 3, 'GK': 2, 'MID': 5}, msg='Team composition incorrect.')
        self.assertListEqual(list(new_team['Name'].values), ['Kelly', 'Alderweireld', 'Ward', 'van Aanholt', 'Alexander-Arnold', 'Origi', 'Abraham', 'Firmino', 'Lloris', 'Ederson', 'Kovacic', 'Son', 'Mount', 'David Silva', 'De Bruyne'])
        self.assertLessEqual(new_team['Current Cost'].sum(), budget)
        self.assertGreaterEqual(new_team['Current Cost'].sum(), budget*0.95)

        out_players = team[lambda df: df['Recommendation'] == 'Transfer out']
        self.assertListEqual(list(out_players['Name'].values),  ['Matip', 'Ayew'])

        in_players = team[lambda df: df['Recommendation'] == 'Transfer in']
        self.assertListEqual(list(in_players['Name'].values), ['van Aanholt', 'Origi'])



    def test_player_formation_invalid(self):
        players = pd.read_csv(self.test_file).set_index('Player Code')
        with self.assertRaisesRegex(ValueError, 'is not a subset of the maximum formation'):
            optimiser.get_optimal_team(players, (2, 6, 5, 3), budget=100.0, optimise_team_on='Expected Points Next 5 GWs', optimise_sel_on='Expected Points Next GW')


    def test_player_df_invalid(self):
        with self.assertRaisesRegex(ValueError, "'Current Cost', 'News And Date', 'Field Position', 'Player Team Code', 'Stats Completeness Percent', 'Expected Points Next 5 GWs', 'Expected Points Next GW', 'Expected Points Next GW'"):
            optimiser.get_optimal_team(pd.DataFrame(), (2, 5, 5, 3), budget=100.0, optimise_team_on='Expected Points Next 5 GWs', optimise_sel_on='Expected Points Next GW')


    def test_player_df_invalid_with_recommend(self):
        with self.assertRaisesRegex(ValueError, "'Current Cost', 'News And Date', 'Field Position', 'Player Team Code', 'Stats Completeness Percent', 'Expected Points Next 5 GWs', 'Expected Points Next GW', 'In Team\?', 'Expected Points Next GW'"):
            optimiser.get_optimal_team(pd.DataFrame(), (2, 5, 5, 3), budget=100.0, optimise_team_on='Expected Points Next 5 GWs', optimise_sel_on='Expected Points Next GW', recommend=2)


    def test_budget_too_low(self):
        players = pd.read_csv(self.test_file).set_index('Player Code')

        budget=70.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with self.assertRaisesRegex(Exception, f'An optimal solution within the max_budget could not be found.'):
                optimiser.get_optimal_team(players, (2, 5, 5, 3), budget=budget, optimise_team_on='Expected Points Next 5 GWs', optimise_sel_on='Expected Points Next GW')


    def test_risk_too_low(self):
        players = pd.read_csv(self.test_file).set_index('Player Code')

        risk=-1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with self.assertRaisesRegex(Exception, f'The risk argument has to be between 0 and 1 but is {risk}.'):
                optimiser.get_optimal_team(players, (2, 5, 5, 3), budget=100.0, optimise_team_on='Expected Points Next 5 GWs', optimise_sel_on='Expected Points Next GW', risk=risk)


    def test_risk_too_high(self):
        players = pd.read_csv(self.test_file).set_index('Player Code')

        risk=2
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with self.assertRaisesRegex(Exception, f'The risk argument has to be between 0 and 1 but is {risk}.'):
                optimiser.get_optimal_team(players, (2, 5, 5, 3), budget=100.0, optimise_team_on='Expected Points Next 5 GWs', optimise_sel_on='Expected Points Next GW', risk=risk)
