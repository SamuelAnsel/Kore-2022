from typing import List
from gym import spaces
import numpy as np
from kore.environment.config import MAX_SHIPYARDS

class ActionsAdapter:
    def __init__(self) -> None:
        self._actions = ['NONE']

    @property
    def action_space(self) -> spaces.Space:
        return spaces.Space

    @property
    def actions(self) -> List[str]:
        return self._actions

class BasicMiningActionsAdapter(ActionsAdapter):
    def __init__(self) -> None:
        self._actions = ['NONE', 'SPAWN', 'MINE']

    @property
    def action_space(self) -> spaces.MultiDiscrete:
        return spaces.MultiDiscrete(
            1 * [len(self.actions)]
        )

class ExpandingActionsAdapter(ActionsAdapter):
    def __init__(self) -> None:
        self._actions = ['NONE', 'SPAWN', 'MINE', 'SHIPYARD']

    @property
    def action_space(self) -> spaces.MultiDiscrete:
        return spaces.MultiDiscrete(
            MAX_SHIPYARDS * [len(self.actions)]
        )