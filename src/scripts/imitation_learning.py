from applications.connect_four.connect_four import ConnectFourState
from applications.connect_four.heuristics import total_consecutive_squares_eval, max_consecutive_squares_eval
from applications.connect_four.models import BasicCNN
from deep_search.nn.deep_heuristic import DeepHeuristic
from deep_search.nn.learning import ImitationLearning
from deep_search.search.agent import RandomAgent
import wandb


# TODO: move params to a yaml config file
num_episodes = 1000
log_wandb = True
project_name = 'DeepSearch for Connect4'
entity = 'michaelbzms'
run_name = 'TD learning from self'


if __name__ == '__main__':
    # use random agents
    p1 = RandomAgent()
    p2 = RandomAgent()

    # init network
    net = BasicCNN()
    print(net)

    # make deep heuristic
    model = DeepHeuristic(
        value_network=net,
        train=True,
        lr=3e-5,
        weight_decay=0.0
    )

    # starting state
    s0 = ConnectFourState()

    if log_wandb:
        wandb.init(
            project=project_name,
            entity=entity,
            reinit=True,
            name=run_name,  # run name
            # group=model_name,  # group name --> hyperparameter tuning on group
            config={
                "learning_rate": model.lr,
                "weight_decay": model.weight_decay,
                "max_batch_size": model.max_batch_size,
                "min_batch_size": model.min_batch_size,
                **model.value_network.get_model_parameters()
            })

    # imitation learning object
    try:
        il = ImitationLearning(
            agent1=p1,
            agent2=p2,
            student=model,
            teacher=model,    # learn from this heuristic
            minimax_depth=2,
            output_student_file='../../models/test_student.pt',
            wandb=wandb if log_wandb else None
        )
        il.play_episodes(starting_state=s0, num_episodes=num_episodes)
    except Exception as e:
        if log_wandb:
            wandb.finish()
        print('SOMETHING HAPPENED')
        raise e

    if log_wandb:
        wandb.finish()
