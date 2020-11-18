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

    def __assert_team(self, squad, budget):
        self.assertEqual(squad.shape[0], 15, msg='Row count must be 15.')

        position_counts = squad[['Field Position', 'Name']].groupby('Field Position').count()['Name'].to_dict()
        self.assertDictEqual(position_counts, {'DEF': 5, 'FWD': 3, 'GK': 2, 'MID': 5}, msg='Team composition incorrect.')

        self.assertLessEqual(squad['Current Cost'].sum()-0.01, budget, msg='Budget breached.')
        self.assertGreaterEqual(squad['Current Cost'].sum(), budget*0.95, msg='Budget not used well enough.')

        self.assertEqual(squad[squad['Captain?'] == True]['Captain?'].count(), 1, msg='More than one Captain')
        self.assertEqual(squad[squad['Vice Captain?'] == True]['Vice Captain?'].count(), 1, msg='More than one Vice Captain')
        self.assertEqual(squad['Selected?'].isin([True]).sum(), 11, msg='Not 11 players selected.')

        self.assertTrue(squad[(squad['Selected?'] == True) & (squad['Captain?'] == False)]['Point Factor'].eq(1).all())
        self.assertTrue(squad[(squad['Selected?'] == True) & (squad['Captain?'] == True)]['Point Factor'].eq(2).all())
        self.assertTrue(squad[squad['Selected?'] == False]['Point Factor'].eq(0).all())

    def test_optimal_team_high_risk(self):
        players = pd.read_csv(self.test_file).set_index('Player Code')

        budget=100.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            squad = optimiser.get_optimal_squad(players, '2-5-5-3', budget=budget, optimise_team_on='Expected Points Next 5 GWs', optimise_sel_on='Expected Points Next GW', risk=1)

        self.__assert_team(squad, budget)

        self.assertListEqual(list(squad['Name'].values), ['Kelly', 'Alderweireld', 'Matip', 'Ward', 'Alexander-Arnold', 'Ayew', 'Abraham', 'Firmino', 'Lloris', 'Ederson', 'Kovacic', 'Son', 'Mount', 'David Silva', 'De Bruyne'])
        self.assertTrue(squad[squad['Name'] == 'De Bruyne']['Captain?'].iloc[0])
        self.assertTrue(squad[squad['Name'] == 'Abraham']['Vice Captain?'].iloc[0])

        self.assertListEqual(list(squad[squad['Selected?'] == True]['Name'].values),  ['Matip', 'Ward', 'Alexander-Arnold', 'Ayew', 'Abraham', 'Firmino', 'Ederson', 'Son', 'Mount', 'David Silva', 'De Bruyne'])

    def test_optimal_team_low_risk(self):
        players = pd.read_csv(self.test_file).set_index('Player Code')

        budget=100.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            squad = optimiser.get_optimal_squad(players, '2-5-5-3', budget=budget, optimise_team_on='Expected Points Next 5 GWs', optimise_sel_on='Expected Points Next GW', risk=0)

        self.__assert_team(squad, budget)

        self.assertListEqual(list(squad['Name'].values), ['Kelly', 'Alderweireld', 'Ward', 'van Aanholt', 'Alexander-Arnold', 'Origi', 'Abraham', 'Firmino', 'Lloris', 'Ederson', 'Kovacic', 'Son', 'Mount', 'David Silva', 'De Bruyne'])
        self.assertTrue(squad[squad['Name'] == 'De Bruyne']['Captain?'].iloc[0])
        self.assertTrue(squad[squad['Name'] == 'Abraham']['Vice Captain?'].iloc[0])

        self.assertListEqual(list(squad[squad['Selected?'] == True]['Name'].values),  ['Alderweireld', 'Ward', 'van Aanholt', 'Alexander-Arnold', 'Abraham', 'Firmino', 'Ederson', 'Son', 'Mount', 'David Silva', 'De Bruyne'])


    def test_optimal_team_include(self):
        players = pd.read_csv(self.test_file).set_index('Player Code')

        budget=100.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            squad = optimiser.get_optimal_squad(players, '2-5-5-3', budget=budget, optimise_team_on='Expected Points Next 5 GWs', optimise_sel_on='Expected Points Next GW', include=['Mané'], risk=1)

        self.__assert_team(squad, budget)

        self.assertListEqual(list(squad['Name'].values), ['Rose', 'Kelly', 'Alderweireld', 'Matip', 'Ward', 'Ayew', 'Abraham', 'Firmino', 'Lloris', 'Ederson', 'Kovacic', 'Mount', 'David Silva', 'Mané', 'De Bruyne'])

    def test_optimal_team_include_invalid_name(self):
        players = pd.read_csv(self.test_file).set_index('Player Code')

        with self.assertRaisesRegex(ValueError, r'\[\'Marc Maier\'\] in include is not in the Name column of players_df.'):
            optimiser.get_optimal_squad(players, '2-5-5-3', budget=100.0, optimise_team_on='Expected Points Next 5 GWs', optimise_sel_on='Expected Points Next GW', include=['Marc Maier'])


    def test_optimal_team_exclude(self):
        players = pd.read_csv(self.test_file).set_index('Player Code')

        budget=100.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            squad = optimiser.get_optimal_squad(players, '2-5-5-3', budget=budget, optimise_team_on='Expected Points Next 5 GWs', optimise_sel_on='Expected Points Next GW', exclude=['Ederson'], risk=1)

        self.__assert_team(squad, budget)
        self.assertListEqual(list(squad['Name'].values), ['Alderweireld', 'van Dijk', 'Ward', 'van Aanholt', 'Alexander-Arnold', 'Kane', 'Ayew', 'Abraham', 'Adrián', 'Lloris', 'Kovacic', 'Mount', 'Bernardo Silva', 'David Silva', 'De Bruyne'])


    def test_optimal_team_include_exclude(self):
        players = pd.read_csv(self.test_file).set_index('Player Code')

        budget=100.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            squad = optimiser.get_optimal_squad(players, '2-5-5-3', budget=budget, optimise_team_on='Expected Points Next 5 GWs', optimise_sel_on='Expected Points Next GW', include=['Mané'], exclude=['Ederson'], risk=1)

        self.__assert_team(squad, budget)
        self.assertListEqual(list(squad['Name'].values), ['Kelly', 'Alderweireld', 'Walker', 'Ward', 'Alexander-Arnold', 'Ayew', 'Abraham', 'Firmino', 'Whiteman', 'Lloris', 'Kovacic', 'Mount', 'David Silva', 'Mané', 'De Bruyne'])


    def test_optimal_team_exclude_invalid_name(self):
        players = pd.read_csv(self.test_file).set_index('Player Code')

        with self.assertRaisesRegex(ValueError, r'\[\'Marc Maier\'\] in exclude is not in the Name column of players_df.'):
            optimiser.get_optimal_squad(players, '2-5-5-3', budget=100.0, optimise_team_on='Expected Points Next 5 GWs', optimise_sel_on='Expected Points Next GW', exclude=['Marc Maier'])


    def test_optimal_team_nan_in_cost(self):
        players = pd.read_csv(self.test_file).set_index('Player Code')
        players.at[526, 'Current Cost'] = np.nan

        with self.assertRaisesRegex(ValueError, r'At least on of the entries in the Current Cost column of the players_df data frame is NaN. The Current Cost must be set for all players.'):
            optimiser.get_optimal_squad(players, '2-5-5-3', budget=100.0, optimise_team_on='Expected Points Next 5 GWs', optimise_sel_on='Expected Points Next GW')


    def test_optimal_team_nan_in_optimise_team_on(self):
        players = pd.read_csv(self.test_file).set_index('Player Code')
        players.at[526, 'Expected Points Next 5 GWs'] = np.nan

        with self.assertRaisesRegex(ValueError, r'At least on of the entries in the Expected Points Next 5 GWs column of the players_df data frame is NaN. The Expected Points Next 5 GWs must be set for all players.'):
            optimiser.get_optimal_squad(players, '2-5-5-3', budget=100.0, optimise_team_on='Expected Points Next 5 GWs', optimise_sel_on='Expected Points Next GW')


    def test_optimal_team_nan_in_optimise_sel_on(self):
        players = pd.read_csv(self.test_file).set_index('Player Code')
        players.at[526, 'Expected Points Next GW'] = np.nan

        with self.assertRaisesRegex(ValueError, r'At least on of the entries in the Expected Points Next GW column of the players_df data frame is NaN. The Expected Points Next GW must be set for all players.'):
            optimiser.get_optimal_squad(players, '2-5-5-3', budget=100.0, optimise_team_on='Expected Points Next 5 GWs', optimise_sel_on='Expected Points Next GW')


    def test_recommend_transfer(self):
        players = pd.read_csv(self.test_file).set_index('Player Code')

        budget = 100.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            squad = optimiser.get_optimal_squad(players, '2-5-5-3', budget=budget, optimise_team_on='Expected Points Next 5 GWs', optimise_sel_on='Expected Points Next GW',  recommend=2)

        self.assertEqual(squad.shape[0], 15, msg='Row count must be 15.')

        position_counts = squad[['Field Position', 'Name']].groupby('Field Position').count()['Name'].to_dict()
        self.assertDictEqual(position_counts, {'DEF': 5, 'FWD': 3, 'GK': 2, 'MID': 5}, msg='Team composition incorrect.')
        self.assertListEqual(list(squad['Name'].values), ['Kelly', 'Alderweireld', 'Ward', 'van Aanholt', 'Alexander-Arnold', 'Origi', 'Abraham', 'Firmino', 'Lloris', 'Ederson', 'Kovacic', 'Son', 'Mount', 'David Silva', 'De Bruyne'])
        self.assertLessEqual(squad['Current Cost'].sum(), budget)
        self.assertGreaterEqual(squad['Current Cost'].sum(), budget*0.95)


    def test_player_df_invalid(self):
        with self.assertRaisesRegex(ValueError, "'Current Cost', 'News And Date', 'Field Position', 'Player Team Code', 'Stats Completeness Percent', 'Expected Points Next 5 GWs', 'Expected Points Next GW', 'Expected Points Next GW'"):
            optimiser.get_optimal_squad(pd.DataFrame(), '2-5-5-3', budget=100.0, optimise_team_on='Expected Points Next 5 GWs', optimise_sel_on='Expected Points Next GW')


    def test_player_df_invalid_with_recommend(self):
        with self.assertRaisesRegex(ValueError, "'Current Cost', 'News And Date', 'Field Position', 'Player Team Code', 'Stats Completeness Percent', 'Expected Points Next 5 GWs', 'Expected Points Next GW', 'In Team\?', 'Expected Points Next GW'"):
            optimiser.get_optimal_squad(pd.DataFrame(), '2-5-5-3', budget=100.0, optimise_team_on='Expected Points Next 5 GWs', optimise_sel_on='Expected Points Next GW', recommend=2)


    def test_budget_too_low(self):
        players = pd.read_csv(self.test_file).set_index('Player Code')

        budget=70.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with self.assertRaisesRegex(Exception, f'An optimal solution within the budget {budget} could not be found.'):
                optimiser.get_optimal_squad(players, '2-5-5-3', budget=budget, optimise_team_on='Expected Points Next 5 GWs', optimise_sel_on='Expected Points Next GW')


    def test_risk_too_low(self):
        players = pd.read_csv(self.test_file).set_index('Player Code')

        risk=-1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with self.assertRaisesRegex(Exception, f'The risk argument has to be between 0 and 1 but is {risk}.'):
                optimiser.get_optimal_squad(players, '2-5-5-3', budget=100.0, optimise_team_on='Expected Points Next 5 GWs', optimise_sel_on='Expected Points Next GW', risk=risk)


    def test_risk_too_high(self):
        players = pd.read_csv(self.test_file).set_index('Player Code')

        risk=2
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with self.assertRaisesRegex(Exception, f'The risk argument has to be between 0 and 1 but is {risk}.'):
                optimiser.get_optimal_squad(players, '2-5-5-3', budget=100.0, optimise_team_on='Expected Points Next 5 GWs', optimise_sel_on='Expected Points Next GW', risk=risk)
