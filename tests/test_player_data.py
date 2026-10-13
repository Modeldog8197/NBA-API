from nba_app.player_data import as_float, as_int, seasons_to_fetch


def test_numeric_cleaning():
    assert as_int("12") == 12
    assert as_int("12.0") == 12
    assert as_int(None) is None
    assert as_float("3.5") == 3.5
    assert as_float("nan") is None


def test_season_window_has_ten_seasons():
    seasons = seasons_to_fetch(10)
    assert len(seasons) == 10
    assert all(len(season) == 7 and season[4] == "-" for season in seasons)
