import unittest, os
from pandas.util.testing import assert_frame_equal
from jupyter import *


class TestJupyterUtils(unittest.TestCase):
    test_file = os.path.join(os.path.dirname(__file__), 'test_data_optimiser.csv')

    def test_summarise_team(self):
        players = pd.read_csv(self.test_file).set_index('Player ID')
        players = players[players['In Team?'] == True]
        expected =  [{'Team': 'All Players', 'Current Cost': '99.3', 'Expected Points Next GW': '98.9', 'Expected Points Next 5 GWs': '480', 'Expected Points GWs To Half': '890.1', 'Expected Points GWs To End': '2769.2', 'Fixture Point Consistency': '28.4' },
                     {'Team': 'Selected Players', 'Current Cost': '80.1', 'Expected Points Next GW': '80.0', 'Expected Points Next 5 GWs': '400', 'Expected Points GWs To Half': '720', 'Expected Points GWs To End': '2240.0', 'Fixture Point Consistency': '31.45' }]

        expected_df = pd.DataFrame.from_dict(expected).set_index('Team').astype('float')
        actual_df = summarise_team(players)
        assert_frame_equal(expected_df, actual_df, check_dtype=False, check_exact=False, check_less_precise=True)
