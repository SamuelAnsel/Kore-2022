import numpy as np
from kaggle_environments import make

# Read env specification
ENV = make('kore_fleets')
ENV_SPECIFICATION = ENV.specification
SHIP_COST = ENV_SPECIFICATION.configuration.spawnCost.default
SHIPYARD_COST = ENV_SPECIFICATION.configuration.convertCost.default
GAME_CONFIG = ENV.configuration

DTYPE = np.float64
MAX_OBSERVABLE_KORE = 500
MAX_OBSERVABLE_SHIPS = 200
MAX_ACTION_FLEET_SIZE = 150
MAX_KORE_IN_RESERVE = 40000
WIN_REWARD = 1000
MAX_KORE = 10000
MAX_SHIPS = 2000
MAX_FLEETS = 441 - 2
MAX_SHIPYARDS = 25
MAX_SPAWN = 10