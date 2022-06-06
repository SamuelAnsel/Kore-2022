from kaggle_environments.envs.kore_fleets.helpers import *
import numpy as np
from kore.utils import Point


def find_shipyard_spot(launchning_shipyard: Point, other_shipyards: List[Point], kore_map: np.ndarray, on_radius=5, weight_radius=10) -> List[Point]:
    bcx, bcy = kore_map.shape[0] // 2, kore_map.shape[1] // 2

    x = np.arange(0, 21)
    y = np.arange(0, 21)

    # Shipyard center
    cx = launchning_shipyard.x
    cy = launchning_shipyard.y
    ox = bcx - cx
    oy = bcy - cy

    # All positions will be checked => At a strict distance of 5 of the shipyard
    circle_pos = abs(x[np.newaxis,:]-bcx) + abs(y[:,np.newaxis]-bcy) == on_radius
    circle_pos = np.roll(circle_pos, -ox, axis=0)
    circle_pos = np.roll(circle_pos, -oy, axis=1)

    board_pos = kore_map.copy()
    board_pos[circle_pos] = np.max(kore_map) * 1.5

    scores = list()
    pos = np.where(circle_pos)

    # Check each potential point
    for px, py in zip(pos[0], pos[1]):
        kore_weights = np.zeros(kore_map.shape)

        ox = bcx - px
        oy = bcy - py

        # Make a filter according to distance
        for r in range(1, weight_radius):
            mask = abs(x[np.newaxis,:]-bcx) + abs(y[:,np.newaxis]-bcy) == r
            kore_weights[mask] = 1 / r

        kore_weights = np.roll(kore_weights, -ox, axis=0)
        kore_weights = np.roll(kore_weights, -oy, axis=1)

        # Weighted sum on nearby kore to the get score
        scores += [[Point(px, py), np.sum(kore_weights * kore_map)]]

    points = list()
    # sort spots and keep only if no shipyard are on it
    sorted_points = [x[0] for x in sorted(scores, key=lambda x: x[1])[::-1] if x[0] not in other_shipyards]

    # only keep points if their position keeps a given distance with other shipyards
    for p in sorted_points:
        keep = True

        for q in other_shipyards:
            if p.distance_to(q, 21) <= 5:
                keep = False
                break

        if keep:
            points += [p]

    # if the previous condition was not match, well take the best point
    if len(points) == 0:
        points = sorted_points

    return points[0]