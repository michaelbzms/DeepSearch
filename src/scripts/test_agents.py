from applications.connect_four.connect_four import ConnectFourState, ConnectFourAction
from deep_search.search.agent import AlphaBetaAgent
from deep_search.search.episode import TwoPlayerGameEpisode
from scripts.test_connect4 import simple_connect4_heuristic


if __name__ == '__main__':
    # start state
    start = ConnectFourState()

    # player 1
    player1 = AlphaBetaAgent(depth=3, player=1, heuristic=simple_connect4_heuristic)

    # player 2
    player2 = AlphaBetaAgent(depth=1, player=2, heuristic=lambda x: 0.0)

    # episode
    episode = TwoPlayerGameEpisode(start, player1, player2)
    episode.play_episode()
    print('Winner is:', episode.get_winner())
