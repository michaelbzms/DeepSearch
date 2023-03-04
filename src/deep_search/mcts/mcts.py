import numpy as np
from numba.experimental import jitclass
from numba import int32, float32

from deep_search.search.state import GameState


class MCTS:
    def __init__(self, start_state: GameState):
        self.root = NodeWrapper(start_state)


class NodeWrapper:
    def __init__(self, state: GameState):
        self.node = Node(num_actions=state.get_num_possible_actions())
        self.state = state


@jitclass([
    ('v', float32),
    ('visit_counts', int32[:]),
    ('ws', float32[:]),
    ('qs', float32[:]),
    ('priors', float32[:]),
])
class Node(object):
    def __init__(self, num_actions: int):
        self.v = 0.0     # state value
        self.visit_counts = np.zeros(num_actions, dtype=np.int32)      # visit counts
        self.ws = np.zeros(num_actions, dtype=np.float32)              # intermediary value TODO?
        self.qs = np.zeros(num_actions, dtype=np.float32)              # action value
        self.priors = np.ones(num_actions, dtype=np.float32)           # prior probability TODO: add dirichlet noise


if __name__ == '__main__':
    from applications.connect_four.connect_four import ConnectFourState

    start = ConnectFourState()

    tree = MCTS(start)

