import unittest
import pandas as pd
import optimiser
import warnings
import os


class TestOptimiser(unittest.TestCase):
    """
    Unit tests the optimiser functions.
    """

    test_file = os.path.join(os.path.dirname(__file__), 'test_data_optimiser.csv')

    def test_optimal_team(self):
        players = pd.read_csv(self.test_file).set_index('Player ID')

        budget=100.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            squad = optimiser.get_optimal_squad(players, '2-5-5-3', budget=budget, optimise_team_on='Expected Points Next 5 GWs', optimise_sel_on='Expected Points Next GW')

        self.assertEqual(squad.shape[0], 15, msg='Row count must be 15.')

        position_counts = squad[['Field Position', 'Name']].groupby('Field Position').count()['Name'].to_dict()
        self.assertDictEqual(position_counts, {'DEF': 5, 'FWD': 3, 'GK': 2, 'MID': 5}, msg='Team composition incorrect.')

        self.assertListEqual(list(squad['Name'].values), ['Kelly', 'Alderweireld', 'Matip', 'Ward', 'Alexander-Arnold', 'Ayew', 'Abraham', 'Firmino', 'Lloris', 'Ederson', 'Kovacic', 'Son', 'Mount', 'David Silva', 'De Bruyne'])

        self.assertLessEqual(squad['Current Cost'].sum()-0.01, budget, msg='Budget breached.')
        self.assertGreaterEqual(squad['Current Cost'].sum(), budget*0.95, msg='Budget not used well enough.')

        self.assertTrue(squad[squad['Name'] == 'De Bruyne']['Captain?'].iloc[0])
        self.assertTrue(squad[squad['Name'] == 'Abraham']['Vice Captain?'].iloc[0])
        self.assertEqual(squad[squad['Captain?'] == True]['Captain?'].count(), 1, msg='More than one Captain')
        self.assertEqual(squad[squad['Vice Captain?'] == True]['Vice Captain?'].count(), 1, msg='More than one Vice Captain')
        self.assertListEqual(list(squad[squad['Selected?'] == True]['Name'].values),  ['Matip', 'Ward', 'Alexander-Arnold', 'Ayew', 'Abraham', 'Firmino', 'Ederson', 'Son', 'Mount', 'David Silva', 'De Bruyne'])
        self.assertEqual(squad['Selected?'].isin([True]).sum(), 11, msg='Not 11 players selected.')

        self.assertTrue(squad[(squad['Selected?'] == True) & (squad['Captain?'] == False)]['Point Factor'].eq(1).all())
        self.assertTrue(squad[(squad['Selected?'] == True) & (squad['Captain?'] == True)]['Point Factor'].eq(2).all())
        self.assertTrue(squad[squad['Selected?'] == False]['Point Factor'].eq(0).all())

    def test_recommend_transfer(self):
        players = pd.read_csv(self.test_file).set_index('Player ID')

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


    def test_player_df_valid(self):
        with self.assertRaisesRegex(ValueError, "['Current Cost', 'News', 'Field Position', 'Team ID', 'Total Points']"):
            optimiser.get_optimal_squad(pd.DataFrame(), '2-5-5-3', budget=100.0, optimise_team_on='Expected Points Next 5 GWs', optimise_sel_on='Expected Points Next GW')


    def test_player_df_valid_with_recommend(self):
        with self.assertRaisesRegex(ValueError, "['Current Cost', 'News', 'Field Position', 'Team ID', 'Total Points', 'In Team?']"):
            optimiser.get_optimal_squad(pd.DataFrame(), '2-5-5-3', budget=100.0, optimise_team_on='Expected Points Next 5 GWs', optimise_sel_on='Expected Points Next GW', recommend=2)


    def test_budget_too_low(self):
        players = pd.read_csv(self.test_file).set_index('Player ID')

        budget=70.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with self.assertRaisesRegex(Exception, f'An optimal solution within the budget {budget} could not be found.'):
                optimiser.get_optimal_squad(players, '2-5-5-3', budget=budget, optimise_team_on='Expected Points Next 5 GWs', optimise_sel_on='Expected Points Next GW')

