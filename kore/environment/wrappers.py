"""
Thanks to Samuel [qdive] (https://www.kaggle.com/lesamu) for sharing this code
https://www.kaggle.com/code/lesamu/reinforcement-learning-baseline-in-python
"""
import gym
import numpy as np
from kaggle_environments import make
from kaggle_environments.envs.kore_fleets.helpers import Board
from typing import Tuple, Dict
from kore.environment.rewards import get_board_value
from kore.environment.config import (
    GAME_CONFIG,
    WIN_REWARD
)
from kore.agents import Agent

class KoreGymEnv(gym.Env):
    """An openAI-gym env wrapper for kaggle's kore environment. Can be used with stable-baselines3.

    There are three fundamental components to this class which you would want to customize for your own agents:
        The action space is defined by `action_space` and `gym_to_kore_action()`
        The state space (observations) is defined by `state_space` and `obs_as_gym_state()`
        The reward is computed with `compute_reward()`

    Note that the action and state spaces define the inputs and outputs to your model *as numpy arrays*. Use the
    functions mentioned above to translate these arrays into actual kore environment observations and actions.

    The rest is basically boilerplate and makes sure that the kaggle environment plays nicely with stable-baselines3.

    First agent is the one being trained

    Usage:
        >>> from stable_baselines3 import PPO
        >>>
        >>> kore_env = KoreGymEnv()
        >>> model = PPO('MlpPolicy', kore_env, verbose=1)
        >>> model.learn(total_timesteps=100000)
    """

    def __init__(self, agent: Agent, opponent, config=None, debug=None):
        super(KoreGymEnv, self).__init__()

        if not config:
            config = GAME_CONFIG
        if not debug:
            debug = True

        self.env = make("kore_fleets", configuration=config, debug=debug)
        self.config = self.env.configuration
        self.trainer = None
        self.raw_obs = None
        self.previous_obs = None
    
        self.agents = [None, opponent]
        self.trained_agent = agent

        self.strict_reward = config.get('strict', False)

        # Debugging info - Enable or disable as needed
        self.reward = 0
        self.n_steps = 0
        self.n_resets = 0
        self.n_dones = 0
        self.last_action = None
        self.last_done = False
        
        self.observation_space = self.trained_agent.observations.observation_space
        self.action_space = self.trained_agent.action_adapter.action_space

    def reset(self) -> np.ndarray:
        """Resets the trainer and returns the initial observation in state space.

        Returns:
            self.obs_as_gym_state: the current observation encoded as a state in state space
        """
        # agents = self.agents if np.random.rand() > .5 else self.agents[::-1]  # Randomize starting position
        self.trainer = self.env.train(self.agents)
        self.raw_obs = self.trainer.reset()
        self.n_resets += 1
        self.trained_agent.observations.update(self.raw_obs)
        observations = self.trained_agent.observations.as_features()
        
        return observations

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute action in the trainer and return the results.

        Args:
            action: The action in action space, i.e. the output of the stable-baselines3 agent

        Returns:
            self.obs_as_gym_state: the current observation encoded as a state in state space
            reward: The agent's reward
            done: If True, the episode is over
            info: A dictionary with additional debugging information
        """
        kore_action = self.trained_agent.action_to_kore_action(action)
        self.previous_obs = self.raw_obs
        self.raw_obs, _, done, info = self.trainer.step(kore_action)  # Ignore trainer reward, which is just delta kore
        self.trained_agent.observations.update(self.raw_obs)
        observations = self.trained_agent.observations.as_features()
        self.reward = self.compute_reward(done)

        # Debugging info
        # with open('logs/tmp.log', 'a') as log:
        #    print(kore_action.action_type, kore_action.num_ships, kore_action.flight_plan, file=log)
        #    if done:
        #        print('done', file=log)
        #    if info:
        #        print('info', file=log)

        self.n_steps += 1
        self.last_done = done
        self.last_action = kore_action
        self.n_dones += 1 if done else 0
        
        return observations, self.reward, done, info

    def render(self, **kwargs):
        self.env.render(**kwargs)

    def close(self):
        pass

    @property
    def board(self):
        return Board(self.raw_obs, self.config)

    @property
    def previous_board(self):
        return Board(self.previous_obs, self.config)

    def compute_reward(self, done: bool, strict=False) -> float:
        """Compute the agent reward. Welcome to the fine art of RL.

         We'll compute the reward as the current board value and a final bonus if the episode is over. If the player
          wins the episode, we'll add a final bonus that increases with shorter time-to-victory.
        If the player loses, we'll subtract that bonus.

        Args:
            done: True if the episode is over
            strict: If True, count only wins/loses (Useful for evaluating a trained agent)

        Returns:
            The agent's reward
        """
        board = self.board
        previous_board = self.previous_board

        if strict:
            if done:
                # Who won?
                # Ugly but 99% sure correct, see https://www.kaggle.com/competitions/kore-2022/discussion/324150#1789804
                agent_reward = self.raw_obs.players[0][0]
                opponent_reward = self.raw_obs.players[1][0]
                return int(agent_reward > opponent_reward)
            else:
                return 0
        else:
            if done:
                # Who won?
                agent_reward = self.raw_obs.players[0][0]
                opponent_reward = self.raw_obs.players[1][0]
                if agent_reward is None or opponent_reward is None:
                    we_won = -1
                else:
                    we_won = 1 if agent_reward > opponent_reward else -1
                win_reward = we_won * (WIN_REWARD + 5 * (GAME_CONFIG['episodeSteps'] - board.step))
            else:
                win_reward = 0

            return get_board_value(board) - get_board_value(previous_board) + win_reward
