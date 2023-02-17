from applications.connect_four.connect_four import ConnectFourState
from applications.connect_four.heuristics import total_consecutive_squares_eval
from applications.connect_four.models import BasicCNN
from deep_search.nn.deep_heuristic import DeepHeuristic
from deep_search.nn.learning import ImitationLearning
from deep_search.search.agent import RandomAgent


# TODO: move params to a yaml config file
num_episodes = 100


if __name__ == '__main__':
    # use random agents
    p1 = RandomAgent()
    p2 = RandomAgent()

    # init network
    net = BasicCNN()   # TODO
    print(net)

    # make deep heuristic
    model = DeepHeuristic(
        value_network=net,
        train=True,
        lr=3e-4,
        weight_decay=1e-6
    )

    # starting state
    s0 = ConnectFourState()

    # imitation learning object
    il = ImitationLearning(
        agent1=p1,
        agent2=p2,
        student=model,
        teacher=total_consecutive_squares_eval,    # learn from this heuristic
        minimax_depth=2,
        output_student_file='../../models/test_student.pt'
    )
    il.play_episodes(starting_state=s0, num_episodes=num_episodes)
