import unittest, os
from pandas.util.testing import assert_frame_equal
from data import *

class TestCalcEpsForNextGws(unittest.TestCase):
    """
    Unit tests the calc_eps_for_next_gws function.
    """

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


class TestGetNextGwCounts(unittest.TestCase):
    """
    Unit tests the get_next_gw_counts function.
    """
    def test_get_next_gw_counts_first_gw(self):
        next_gw_counts = get_next_gw_counts(['Next GW', 'Next 7 GWs', 'GWs To End'], 1, 38)
        self.assertDictEqual(next_gw_counts, {'Next GW': 1, 'Next 7 GWs': 7, 'GWs To End': 38})

    def test_get_next_gw_counts_last_gw(self):
        next_gw_counts = get_next_gw_counts(['Next GW', 'Next 7 GWs', 'GWs To End'], 38, 38)
        self.assertDictEqual(next_gw_counts, {'Next GW': 1, 'Next 7 GWs': 1, 'GWs To End': 1})

    def test_get_next_gw_counts_after_last_gw(self):
        next_gw_counts = get_next_gw_counts(['Next GW', 'Next 7 GWs', 'GWs To End'], 39, 38)
        self.assertDictEqual(next_gw_counts, {'Next GW': 0, 'Next 7 GWs': 0, 'GWs To End': 0})


class TestCalcEps(unittest.TestCase):
    """
    Unit tests the get_next_gw_counts function.
    """
    test_file = os.path.join(os.path.dirname(__file__), 'test_calc_eps.csv')

    def test_calc_eps(self):
        players_fixture_team_eps = (pd.read_csv(self.test_file)
                                    .assign(**{'Expected Points Calc': lambda df: df.pipe(calc_eps)})
                                    .reset_index(drop=True))

        assert_frame_equal(pd.read_csv('assert_calc_eps.csv'), players_fixture_team_eps)


class TestCalcPlayerFixtureStats(unittest.TestCase):
    """
    Unit tests the get_next_gw_counts function.
    """
    test_file = os.path.join(os.path.dirname(__file__), 'test_calc_player_fixture_stats.csv')

    def test_calc_player_fixture_stats(self):
        player_fixture_stats = (pd.read_csv(self.test_file)
                                .set_index(['Player ID', 'Fixture ID'])
                                .pipe(calc_player_fixture_stats)
                                .reset_index())

        assert_frame_equal(pd.read_csv('assert_calc_player_fixture_stats.csv'),
                           player_fixture_stats,
                           check_dtype=False)


class TestProjToGw(unittest.TestCase):
    """
    Unit tests the proj_to_gw function.
    """
    test_file = os.path.join(os.path.dirname(__file__), 'test_proj_to_gw.csv')

    def test_proj_to_gw_for_normal_gw(self):
        players_gw_team_eps = (pd.read_csv(self.test_file)
                                .set_index(['Player ID', 'Fixture ID'])
                                .pipe(proj_to_gw)
                                .reset_index()
                                [lambda df: df['Game Week'] == 17]
                                .set_index(['Player ID', 'Game Week'])
                               [['Name', 'Expected Points', 'Fixture Short Name Difficulty', 'Rel. Fixture Strength']])

        assert_frame_equal(pd.read_csv('assert_proj_to_gw_for_normal_gw.csv').set_index(['Player ID', 'Game Week']),
                       players_gw_team_eps,
                       check_dtype=False)

    def test_proj_to_gw_for_missing_gw(self):
        players_gw_team_eps = (pd.read_csv(self.test_file)
                                .set_index(['Player ID', 'Fixture ID'])
                                .pipe(proj_to_gw)
                                .reset_index()
                                [lambda df: df['Game Week'] == 18]
                                .set_index(['Player ID', 'Game Week'])
                               [['Name', 'Expected Points', 'Fixture Short Name Difficulty', 'Rel. Fixture Strength']])

        assert_frame_equal(pd.read_csv('assert_proj_to_gw_for_missing_gw.csv').set_index(['Player ID', 'Game Week']),
                       players_gw_team_eps,
                       check_dtype=False)


    def test_proj_to_gw_for_double_gw(self):
        players_gw_team_eps = (pd.read_csv(self.test_file)
                                .set_index(['Player ID', 'Fixture ID'])
                                .pipe(proj_to_gw)
                                .reset_index()
                                [lambda df: df['Game Week'] == 24]
                                .set_index(['Player ID', 'Game Week'])
                               [['Name', 'Expected Points', 'Fixture Short Name Difficulty', 'Rel. Fixture Strength']])

        assert_frame_equal(pd.read_csv('assert_proj_to_gw_for_double_gw.csv').set_index(['Player ID', 'Game Week']),
                       players_gw_team_eps,
                       check_dtype=False)