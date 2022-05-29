from kaggle_environments.envs.kore_fleets.helpers import *
import random
import numpy as np
from kore.utils import max_fp_len
from kore.mining import get_shipyard_mining_walks, Walk

class Agent:
    def __init__(self):
        pass

    def next_action(self, obs, config):
        pass

class RandomMiningAgent(Agent):
    def __init__(self):
        super().__init__()
        self.ACTIONS = ['MINE', 'SPAWN']

    def next_action(self, obs, config):
        board = Board(obs, config)
        me=board.current_player
        me = board.current_player
        kore_map = np.flip(np.array(obs['kore']).reshape((21,21)).transpose(), 1)
        turn = board.step
        spawn_cost = board.configuration.spawn_cost
        kore_left = me.kore
        
        # loop through all shipyards you control
        for shipyard in me.shipyards:
            macro_action = random.choice(self.ACTIONS)

            if macro_action == 'SPAWN' and kore_left >= spawn_cost:
                n_ships = int(min(kore_left // spawn_cost, shipyard.max_spawn))
                action = ShipyardAction.spawn_ships(n_ships)
            elif macro_action == 'MINE' and shipyard.ship_count > 6:
                mining_walks = get_shipyard_mining_walks(shipyard.position, kore_map)
                ship_count = random.randint(3, shipyard.ship_count // 2 + 1)
                fp_len = int(min(max_fp_len(ship_count), 7))
                flight_plan = random.choice(mining_walks[fp_len][:10]).flight_plan
                action = ShipyardAction.launch_fleet_with_flight_plan(ship_count, flight_plan)
            else:
                action = ShipyardAction.spawn_ships(1)

            shipyard.next_action = action

        return me.next_actions