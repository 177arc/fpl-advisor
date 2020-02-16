import unittest, os
from pandas.util.testing import assert_frame_equal
from data import *

# Unit tests the calc_eps_for_next_gws function.
class TestCalcEpsForNextGws(unittest.TestCase):
    def test_calc_eps_for_next_gws_first_gw(self):
        player_gw_next_eps = (pd.read_csv('test_calc_eps_for_next_gws.csv')
                              .set_index('Player ID')
                              .groupby('Player ID')
                              .apply(lambda df: df.pipe(calc_eps_for_next_gws, ['Next GW', 'Next 3 GWs', 'GWs To End'], 1, 38))
                              .reset_index())

        assert_frame_equal(pd.read_csv('assert_calc_eps_for_next_gws_first_gw.csv'), player_gw_next_eps, check_dtype=False)

    def test_calc_eps_for_next_gws_missing_fixture(self):
        player_gw_next_eps = (pd.read_csv('test_calc_eps_for_next_gws.csv')
                              .set_index('Player ID')
                              .groupby('Player ID')
                              .apply(lambda df: df.pipe(calc_eps_for_next_gws, ['Next GW', 'Next 3 GWs', 'GWs To End'], 18, 38))
                              .reset_index())

        assert_frame_equal(pd.read_csv('assert_calc_eps_for_next_gws_missing_fixture.csv'), player_gw_next_eps, check_dtype=False)

    def test_calc_eps_for_next_gws_double_fixture(self):
        player_gw_next_eps = (pd.read_csv('test_calc_eps_for_next_gws.csv')
                              .set_index('Player ID')
                              .groupby('Player ID')
                              .apply(lambda df: df.pipe(calc_eps_for_next_gws, ['Next GW', 'Next 3 GWs', 'GWs To End'], 24, 38))
                              .reset_index())

        assert_frame_equal(pd.read_csv('assert_calc_eps_for_next_gws_double_fixture.csv'), player_gw_next_eps, check_dtype=False)

    def test_calc_eps_for_next_gws_last_fixture(self):
        player_gw_next_eps = (pd.read_csv('test_calc_eps_for_next_gws.csv')
                              .set_index('Player ID')
                              .groupby('Player ID')
                              .apply(lambda df: df.pipe(calc_eps_for_next_gws, ['Next GW', 'Next 3 GWs', 'GWs To End'], 38, 38))
                              .reset_index())

        assert_frame_equal(pd.read_csv('assert_calc_eps_for_next_gws_last_fixture.csv'), player_gw_next_eps, check_dtype=False)

# Unit tests the get_next_gw_counts function.
class TestGetNextGwCounts(unittest.TestCase):
    def test_get_next_gw_counts_first_gw(self):
        next_gw_counts = get_next_gw_counts(['Next GW', 'Next 7 GWs', 'GWs To End'], 1, 38)
        self.assertDictEqual(next_gw_counts, {'Next GW': 1, 'Next 7 GWs': 7, 'GWs To End': 38})

    def test_get_next_gw_counts_last_gw(self):
        next_gw_counts = get_next_gw_counts(['Next GW', 'Next 7 GWs', 'GWs To End'], 38, 38)
        self.assertDictEqual(next_gw_counts, {'Next GW': 1, 'Next 7 GWs': 1, 'GWs To End': 1})

    def test_get_next_gw_counts_after_last_gw(self):
        next_gw_counts = get_next_gw_counts(['Next GW', 'Next 7 GWs', 'GWs To End'], 39, 38)
        self.assertDictEqual(next_gw_counts, {'Next GW': 0, 'Next 7 GWs': 0, 'GWs To End': 0})


# Unit tests the get_next_gw_counts function.
class TestCalcEps(unittest.TestCase):
    test_file = os.path.join(os.path.dirname(__file__), 'test_calc_eps.csv')

    def test_calc_eps(self):
        players_fixture_team_eps = (pd.read_csv(self.test_file)
                                    .assign(**{'Expected Points Calc': lambda df: df.pipe(calc_eps)})
                                    .reset_index(drop=True))

        assert_frame_equal(pd.read_csv('assert_calc_eps.csv'), players_fixture_team_eps)


# Unit tests the get_next_gw_counts function.
class TestCalcPlayerFixtureStats(unittest.TestCase):
    test_file = os.path.join(os.path.dirname(__file__), 'test_calc_player_fixture_stats.csv')

    def test_calc_player_fixture_stats(self):
        player_fixture_stats = (pd.read_csv(self.test_file)
                                .set_index(['Player ID', 'Fixture ID'])
                                .pipe(calc_player_fixture_stats)
                                .reset_index())

        assert_frame_equal(pd.read_csv('assert_calc_player_fixture_stats.csv'),
                           player_fixture_stats,
                           check_dtype=False)