from typing import Any, Dict
from kaggle_environments.envs.kore_fleets.helpers import Board
from gym import spaces
import numpy as np
from kore.environment.config import (
    GAME_CONFIG,
    MAX_KORE,
    MAX_FLEETS,
    MAX_SHIPS,
    MAX_SHIPYARDS,
    MAX_SPAWN
)
from kore.utils import Point

def normalize_clip(x, norm, clip_min=0, clip_max=1):
    return max(clip_min, min(clip_max, x / norm))

def shipyard_info(player_id: int, board: Board):
    info = dict({
        'player_shipyard_count': 0,
        'opponent_shipyard_count': 0,
        'player_ships_count': 0,
        'opponent_ships_count': 0
    })

    for sid, shipyard in board.shipyards.items():
        if shipyard.player_id == player_id:
            info['player_shipyard_count'] += 1
            info['player_ships_count'] += shipyard.ship_count
        else:
            info['opponent_shipyard_count'] += 1
            info['opponent_ships_count'] += shipyard.ship_count

    return info

def fleets_info(player_id: int, board: Board):
    info = dict({
        'player_fleet_count': 0,
        'opponent_fleet_count': 0,
        'player_ships_count': 0,
        'opponent_ships_count': 0
    })

    for fid, fleet in board.fleets.items():
        if fleet.player_id == player_id:
            info['player_fleet_count'] += 1
            info['player_ships_count'] += fleet.ship_count
        else:
            info['opponent_fleet_count'] += 1
            info['opponent_ships_count'] += fleet.ship_count

    return info

def kore_around_shipyard(shipyard_pos: Point, kore_map: np.ndarray):
    bcx, bcy = kore_map.shape[0] // 2, kore_map.shape[1] // 2

    # Shipyard center
    cx = shipyard_pos.x
    cy = shipyard_pos.y
    ox = bcx - cx
    oy = bcy - cy

    # All positions will be checked => At a strict distance of 5 of the shipyard
    mask = np.zeros_like(kore_map)
    size = 11
    mask[cx - size//2:cx + size//2, cy - size//2:cy + size//2] = 1
    mask = np.roll(mask, -ox, axis=0)
    mask = np.roll(mask, -oy, axis=1)

    # Weighted sum on nearby kore to the get score
    kore_around = np.sum(mask * kore_map)

    return kore_around

class ObservationAdapter:
    def __init__(self, raw_observation: Dict[str, Any]) -> None:
        pass

    def as_features(self):
        return

    def observation_space(self) -> spaces.Space:
        return spaces.Space()


# TODO : get mining rate information => from agent (but for the opponent ?) ? from board ? from simulator ?
class GeneralObservationAdapter(ObservationAdapter):
    def __init__(self) -> None:
        pass
    
    def update(self, raw_observation: Dict[str, Any]) -> None:
        self.board = Board(raw_observation, GAME_CONFIG)
        self.turn = self.board.step

        self.player_id = self.board.current_player_id
        self.opponent_id = self.board.opponents[0].id

        sinfo = shipyard_info(self.player_id, self.board)
        self.player_shipyard_count = sinfo['player_shipyard_count']
        self.opponent_shipyard_count = sinfo['opponent_shipyard_count']

        finfo = fleets_info(self.player_id, self.board)
        self.player_fleet_count = finfo['player_fleet_count']
        self.opponent_fleet_count = finfo['opponent_fleet_count']

        self.player_ships_count = sinfo['player_ships_count'] + finfo['player_ships_count']
        self.opponent_ships_count = sinfo['opponent_ships_count'] + finfo['opponent_ships_count']

        self.player_kore = self.board.players[self.player_id].kore
        self.opponent_kore = self.board.players[self.opponent_id].kore
        
    def as_features(self):
        max_shipyards = MAX_SHIPYARDS
        max_fleets = MAX_FLEETS
        max_ships = MAX_SHIPS
        max_kore = MAX_KORE

        features = [
            normalize_clip(self.turn, 400),
            normalize_clip(self.player_shipyard_count, max_shipyards),
            normalize_clip(self.player_fleet_count, max_fleets),
            normalize_clip(self.player_ships_count, max_ships),
            normalize_clip(self.player_kore, max_kore),
            normalize_clip(self.opponent_shipyard_count, max_shipyards),
            normalize_clip(self.opponent_fleet_count, max_fleets),
            normalize_clip(self.opponent_ships_count, max_ships),
            normalize_clip(self.opponent_kore, max_kore)
        ]

        return features

    @property
    def observation_space(self) -> spaces.Box:
        return spaces.Box(
            low=0.,
            high=1.,
            shape=(9,),
            dtype=np.float64
        )

class LocalObservationAdapter(ObservationAdapter):
    def __init__(self) -> None:
        pass
    
    def update(self, raw_observation: Dict[str, Any]) -> None:
        self.board = Board(raw_observation, GAME_CONFIG)
        kore_map = np.flip(np.array(self.board.observation['kore']).reshape((21,21)).transpose(), 1)

        self.shipyards = list()

        for shipyard in self.board.current_player.shipyards:
            kore_around = kore_around_shipyard(shipyard.position, kore_map)
            ships = shipyard.ship_count
            max_spawn = shipyard.max_spawn

            self.shipyards += [{'kore_around': kore_around, 'ships': ships, 'max_spawn': max_spawn}]
        
    def as_features(self):
        max_ships = MAX_SHIPS
        max_kore_around = (11 * 11 - 1) * 500
        max_spawn = MAX_SPAWN

        features = list()

        for shipyard in self.shipyards:
            features += [
                normalize_clip(shipyard['kore_around'], max_kore_around),
                normalize_clip(shipyard['ships'], max_ships),
                normalize_clip(shipyard['max_spawn'], max_spawn)
            ]

        # Padding
        for _ in range(MAX_SHIPYARDS - len(self.shipyards)):
            features += [-1., -1., -1.]

        return features

    @property
    def observation_space(self) -> spaces.Box:
        return spaces.Box(
            low=-1.,
            high=1.,
            shape=(MAX_SHIPYARDS * 3,),
            dtype=np.float64
        )

class HybridObservationAdapter(ObservationAdapter):
    def __init__(self) -> None:
        self.global_observations = GeneralObservationAdapter()
        self.local_observations = LocalObservationAdapter()
    
    def update(self, raw_observation: Dict[str, Any]) -> None:
        self.global_observations.update(raw_observation)
        self.local_observations.update(raw_observation)
        self.board = self.global_observations.board
        
    def as_features(self):
        return dict({
            'global': self.global_observations.as_features(),
            'local': self.local_observations.as_features()
        })

    @property
    def observation_space(self) -> spaces.Box:
        return spaces.Dict({
            'global': self.global_observations.observation_space,
            'local': self.local_observations.observation_space
        })