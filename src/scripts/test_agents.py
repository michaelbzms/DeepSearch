from applications.connect_four.connect_four import ConnectFourState, ConnectFourAction
from applications.connect_four.heuristics import max_consecutive_squares_eval, total_consecutive_squares_eval
from deep_search.search.agent import AlphaBetaAgent
from deep_search.search.episode import TwoPlayerGameEpisode


if __name__ == '__main__':
    # start state
    start = ConnectFourState()

    # player 1
    player1 = AlphaBetaAgent(depth=3, player=1, heuristic=max_consecutive_squares_eval)

    # player 2
    player2 = AlphaBetaAgent(depth=3, player=2, heuristic=total_consecutive_squares_eval)

    # episode
    episode = TwoPlayerGameEpisode(start, player1, player2)
    episode.play_episode()
    print('Winner is:', episode.get_winner())
