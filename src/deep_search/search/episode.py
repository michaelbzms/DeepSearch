import time
from typing import Iterable

from deep_search.search.agent import Agent, GameAgent
from deep_search.search.state import State, GameState


class Episode:
    """
    A sequence of consecutive states until a final state.
    """
    max_depth: int = 100000

    def __init__(self, starting_state: State, *agents: Iterable[Agent]):
        self.state_history = [starting_state]
        self.finished = False
        self.agents: Iterable[Agent] = agents           # iterable of agents that play in turn

    def add_state(self, state: State):
        if self.finished:
            print('Episode already finished')
            return
        # add state to history
        self.state_history.append(state)
        # check if done
        if self.current_state.is_final():
            self.finished = True

    def play_episode(self, verbose=True, wait_sec: float = 1.0):
        for _ in range(self.max_depth):
            # agent(s) play in turn
            for i, agent in enumerate(self.agents):
                start_t = time.time()
                # decide action
                action = agent.decide_action(self.current_state)
                if action is None:
                    print(self.current_state)
                    raise ValueError('Run out of actions')
                # play action
                new_state: State = self.current_state.get_next_state(action)
                self.add_state(new_state)
                # print
                if verbose:
                    print(f'Player {i + 1} played the move: {action}')
                    # draw state
                    new_state.draw()
                    # wait at least wait_sec amount of time
                    while time.time() - start_t < wait_sec:
                        time.sleep(start_t + wait_sec - time.time())
                # check if done
                if self.finished:
                    return

    def is_finished(self):
        return self.finished

    @property
    def current_state(self) -> State:
        return self.state_history[-1]


class SearchEpisode(Episode):
    """
    An episode for a single agent that looks through a search space.
    """
    def __init__(self, starting_state: State, *agents: Iterable[Agent]):
        super().__init__(starting_state, *agents)
        if len(agents) != 1:
            raise ValueError('We need exactly 1 agent for a search episode.')


class TwoPlayerGameEpisode(Episode):
    """
    An episode for a two-player game.
    """
    def __init__(self, starting_state: GameState, *agents: Iterable[GameAgent]):
        super().__init__(starting_state, *agents)
        if len(agents) != 2:
            raise ValueError('We need exactly 2 agents to play a two-player game.')

    @property
    def current_state(self) -> GameState:
        return self.state_history[-1]

    def get_winner(self) -> any or None:
        return self.current_state.get_winner() if self.is_finished() else None
