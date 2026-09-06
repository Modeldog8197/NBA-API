import math

import pytest

from nba_app.court import ARC_JOIN_Y, describe_shot


@pytest.mark.parametrize("x,y,value,zone", [
    (0, 0, 2, "Restricted Area"), (0, 40, 2, "Restricted Area"),
    (0, 41, 2, "Paint (Non-RA)"), (80, 137.5, 2, "Paint (Non-RA)"),
    (81, 137.5, 2, "Mid-Range"), (0, 237.5, 2, "Mid-Range"),
    (0, 237.51, 3, "Above the Break 3"),
    (220, 0, 2, "Mid-Range"), (220.01, 0, 3, "Right Corner 3"),
    (-220.01, -20, 3, "Left Corner 3"), (220, ARC_JOIN_Y, 2, "Mid-Range"),
    (230, ARC_JOIN_Y, 3, "Right Corner 3"),
    (230, ARC_JOIN_Y + 0.01, 3, "Above the Break 3"),
])
def test_boundaries(x, y, value, zone):
    shot = describe_shot(x, y)
    assert shot["shot_value"] == value
    assert shot["shot_zone"] == zone
    assert shot["distance_ft"] == pytest.approx(math.hypot(x, y) / 10)


def test_same_distance_corner_is_three_top_is_two():
    assert describe_shot(225, 0)["shot_value"] == 3
    assert describe_shot(0, 225)["shot_value"] == 2


@pytest.mark.parametrize("x,y", [(251, 0), (-251, 0), (0, -53), (0, 418), (float("nan"), 0), (0, float("inf"))])
def test_invalid_coordinates(x, y):
    with pytest.raises(ValueError):
        describe_shot(x, y)


def test_conflicting_shot_type_and_distance():
    with pytest.raises(ValueError, match="shot_type conflicts"):
        describe_shot(230, 0, shot_type="2pt")
    with pytest.raises(ValueError, match="shot_distance conflicts"):
        describe_shot(30, 40, shot_distance=50)
    assert describe_shot(30, 40, shot_distance=5.05, shot_type="2PT Field Goal")["distance_ft"] == 5


def test_negative_y_does_not_apply_back_of_circle_as_three_line():
    assert describe_shot(219, -50)["shot_value"] == 2


@pytest.mark.parametrize("angle", [0, 0.1, 0.4, 0.9, -0.5])
def test_arc_points_are_two_and_step_outside_is_three(angle):
    x, y = math.sin(angle) * 237.5, math.cos(angle) * 237.5
    assert describe_shot(x, y)["shot_value"] == 2
    assert describe_shot(x * 1.0001, y * 1.0001)["shot_value"] == 3
