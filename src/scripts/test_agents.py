from applications.connect_four.connect_four import ConnectFourState, ConnectFourAction
from applications.connect_four.heuristics import max_consecutive_squares_eval, total_consecutive_squares_eval
from applications.connect_four.models import BasicCNN
from deep_search.nn.deep_heuristic import DeepHeuristic
from deep_search.nn.util import load_model
from deep_search.search.agent import AlphaBetaAgent, RandomAgent
from deep_search.search.episode import TwoPlayerGameEpisode


if __name__ == '__main__':
    # start state
    start = ConnectFourState()

    # load agent
    net = load_model('../../models/test_best_so_far.pt', BasicCNN)
    print(net)
    dh = DeepHeuristic(value_network=net)

    # player 1
    player1 = AlphaBetaAgent(depth=3, player=1, heuristic=dh, use_tt=True, verbose=True)

    # player 2
    # player2 = AlphaBetaAgent(depth=3, player=2, heuristic=max_consecutive_squares_eval)
    player2 = AlphaBetaAgent(depth=3, player=2, heuristic=total_consecutive_squares_eval, use_tt=True, verbose=True)

    # episode
    episode = TwoPlayerGameEpisode(start, player1, player2)
    episode.play_episode()
    print('Winner is:', episode.get_winner())
