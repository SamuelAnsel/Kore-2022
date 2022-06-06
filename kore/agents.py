from kaggle_environments.envs.kore_fleets.helpers import *
from kaggle_environments.envs.kore_fleets.kore_fleets import get_shortest_flight_path_between
import random
import numpy as np
from kore.utils import max_fp_len
from kore.rules.mining import get_shipyard_mining_walks, Walk
from kore.rules.expand import find_shipyard_spot
from kore.environment.observations import *
from kore.environment.actions import *
from stable_baselines3.ppo import PPO

class Agent:
    def __init__(self):
        pass

    def next_action(self, obs, config):
        pass

    def action_to_kore_action(self, action: np.ndarray) -> ShipyardAction:
        return None

class RandomMiningAgent(Agent):
    def __init__(self):
        super().__init__()
        self.observations = GeneralObservationAdapter()
        self.action_adapter = BasicMiningActionsAdapter()

    def action_to_kore_action(self, action: np.ndarray) -> ShipyardAction:
        board = self.observations.board
        me = board.current_player
        kore_map = np.flip(np.array(self.observations.board.observation['kore']).reshape((21,21)).transpose(), 1)
        turn = board.step
        spawn_cost = board.configuration.spawn_cost
        kore_left = me.kore

        # loop through all shipyards you control
        for shipyard in me.shipyards:
            action = self.action_adapter.actions[random.choice(range(len(self.action_adapter.actions)))]

            if action == 'SPAWN' and kore_left >= spawn_cost:
                n_ships = int(min(kore_left // spawn_cost, shipyard.max_spawn))
                kore_action = ShipyardAction.spawn_ships(n_ships)
            elif action == 'MINE' and shipyard.ship_count > 6:
                mining_walks = get_shipyard_mining_walks(shipyard.position, kore_map)
                ship_count = random.randint(3, shipyard.ship_count // 2 + 1)
                fp_len = int(min(max_fp_len(ship_count), 7))
                flight_plan = random.choice(mining_walks[fp_len][:10]).flight_plan
                kore_action = ShipyardAction.launch_fleet_with_flight_plan(ship_count, flight_plan)
            else:
                kore_action = ShipyardAction.spawn_ships(1)

            shipyard.next_action = kore_action

        return me.next_actions

    def next_action(self, obs, config):
        self.observations.update(obs)
        next_actions = self.action_to_kore_action(None)        

        return next_actions

class BasicMiningAgent(Agent):
    def __init__(self):
        super().__init__()
        self.observations = GeneralObservationAdapter()
        self.action_adapter = BasicMiningActionsAdapter()

    def init_model(self, env):
        self.model = PPO('MlpPolicy', env, verbose=2, tensorboard_log='../logs/', n_steps=1200)

    def load_model(self, path):
        self.model = PPO.load(path)

    def action_to_kore_action(self, action: np.ndarray) -> ShipyardAction:
        board = self.observations.board
        me = board.current_player
        kore_map = np.flip(np.array(self.observations.board.observation['kore']).reshape((21,21)).transpose(), 1)
        turn = board.step
        spawn_cost = board.configuration.spawn_cost
        kore_left = me.kore

        # loop through all shipyards you control
        for shipyard in me.shipyards:
            action = self.action_adapter.actions[action[0][0]]

            if action == 'SPAWN' and kore_left >= spawn_cost:
                n_ships = int(min(kore_left // spawn_cost, shipyard.max_spawn))
                kore_action = ShipyardAction.spawn_ships(n_ships)
            elif action == 'MINE' and shipyard.ship_count > 6:
                mining_walks = get_shipyard_mining_walks(shipyard.position, kore_map)
                ship_count = random.randint(3, shipyard.ship_count // 2 + 1)
                fp_len = int(min(max_fp_len(ship_count), 7))
                flight_plan = random.choice(mining_walks[fp_len][:10]).flight_plan
                kore_action = ShipyardAction.launch_fleet_with_flight_plan(ship_count, flight_plan)
            else:
                kore_action = ShipyardAction.spawn_ships(1)

            shipyard.next_action = kore_action

        return me.next_actions

    def next_action(self, obs, config):
        self.observations.update(obs)
        action = self.model.predict(self.observations.as_features())
        next_actions = self.action_to_kore_action (action)         

        return next_actions

class RandomExpandingAgent(Agent):
    def __init__(self):
        super().__init__()
        self.observations = GeneralObservationAdapter()
        self.action_adapter = ExpandingActionsAdapter()

    def action_to_kore_action(self, action: np.ndarray) -> ShipyardAction:
        board = self.observations.board
        me = board.current_player
        kore_map = np.flip(np.array(self.observations.board.observation['kore']).reshape((21,21)).transpose(), 1)
        turn = board.step
        spawn_cost = board.configuration.spawn_cost
        kore_left = me.kore

        # loop through all shipyards you control
        for shipyard in me.shipyards:
            action = random.choice(self.action_adapter.actions)

            if action == 'SPAWN' and kore_left >= spawn_cost:
                n_ships = int(min(kore_left // spawn_cost, shipyard.max_spawn))
                kore_action = ShipyardAction.spawn_ships(n_ships)
            elif action == 'MINE' and shipyard.ship_count > 6:
                mining_walks = get_shipyard_mining_walks(shipyard.position, kore_map)
                ship_count = random.randint(3, shipyard.ship_count // 2 + 1)
                fp_len = int(min(max_fp_len(ship_count), 7))
                flight_plan = random.choice(mining_walks[fp_len][:10]).flight_plan
                kore_action = ShipyardAction.launch_fleet_with_flight_plan(ship_count, flight_plan)
            elif action == 'SHIPYARD' and shipyard.ship_count >= 50:
                this_shipyard_pos = shipyard.position
                other_shipyards = [s.position for s in me.shipyards if s.position != this_shipyard_pos]
                new_shipyard_point = find_shipyard_spot(shipyard.position, other_shipyards, kore_map)
                flight_plan = get_shortest_flight_path_between(this_shipyard_pos, new_shipyard_point, kore_map.shape[0], trailing_digits=True)
                flight_plan += 'C'
                kore_action = ShipyardAction.launch_fleet_with_flight_plan(50, flight_plan)
            else:
                kore_action = ShipyardAction.spawn_ships(1)

            shipyard.next_action = kore_action

        return me.next_actions

    def next_action(self, obs, config):
        self.observations.update(obs)
        next_actions = self.action_to_kore_action(None)        

        return next_actions

class ExpandingAgent(Agent):
    def __init__(self):
        super().__init__()
        self.observations = HybridObservationAdapter()
        self.action_adapter = ExpandingActionsAdapter()
        self.shipyard_launched = 0

    def init_model(self, env):
        self.model = PPO('MultiInputPolicy', env, verbose=2, tensorboard_log='../logs/', n_steps=1200)

    def save_model(self, path):
        self.model.save(path)

    def load_model(self, path):
        self.model = PPO.load(path)

    def action_to_kore_action(self, action: np.ndarray) -> ShipyardAction:
        board = self.observations.board
        me = board.current_player
        kore_map = np.flip(np.array(self.observations.board.observation['kore']).reshape((21,21)).transpose(), 1)
        turn = board.step
        spawn_cost = board.configuration.spawn_cost
        kore_left = me.kore

        # loop through all shipyards you control
        for i, shipyard in enumerate(me.shipyards):
            shipyard_action = self.action_adapter.actions[action[i]]

            if shipyard_action == 'SPAWN' and kore_left >= spawn_cost:
                n_ships = int(min(kore_left // spawn_cost, shipyard.max_spawn))
                kore_action = ShipyardAction.spawn_ships(n_ships)
            elif shipyard_action == 'MINE' and shipyard.ship_count > 6:
                mining_walks = get_shipyard_mining_walks(shipyard.position, kore_map)
                ship_count = random.randint(3, shipyard.ship_count // 2 + 1)
                fp_len = int(min(max_fp_len(ship_count), 7))
                flight_plan = random.choice(mining_walks[fp_len][:10]).flight_plan
                kore_action = ShipyardAction.launch_fleet_with_flight_plan(ship_count, flight_plan)
            elif shipyard_action == 'SHIPYARD' and shipyard.ship_count >= 50 and self.shipyard_launched < MAX_SHIPYARDS:
                this_shipyard_pos = shipyard.position
                other_shipyards = [s.position for s in me.shipyards if s.position != this_shipyard_pos]
                new_shipyard_point = find_shipyard_spot(shipyard.position, other_shipyards, kore_map)
                flight_plan = get_shortest_flight_path_between(this_shipyard_pos, new_shipyard_point, kore_map.shape[0], trailing_digits=True)
                flight_plan += 'C'
                kore_action = ShipyardAction.launch_fleet_with_flight_plan(50, flight_plan)
                self.shipyard_launched += 1 # TODO : Check if it has been constructed
            else:
                kore_action = ShipyardAction.spawn_ships(1)

            shipyard.next_action = kore_action

        return me.next_actions

    def next_action(self, obs, config):
        if obs['step'] == 0:
            self.shipyard_launched = 0
            
        self.observations.update(obs)
        action = self.model.predict(self.observations.as_features())[0]
        next_actions = self.action_to_kore_action (action)         

        return next_actions