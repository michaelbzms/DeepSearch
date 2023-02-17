from typing import Callable
from tqdm import tqdm

from deep_search.nn.deep_heuristic import DeepHeuristic
from deep_search.search.agent import GameAgent
from deep_search.search.algorithms import GameNode, minimax
from deep_search.search.state import GameState


class ImitationLearning:
    def __init__(self,
                 agent1: GameAgent,
                 agent2: GameAgent,
                 student: DeepHeuristic,
                 teacher: Callable[[GameState], float],
                 minimax_depth: int = 3,
                 output_student_file: str = '../../models/student.pt',
                 max_game_turns=10000):
        """
        Agents should not be deterministic! Their role is to dynamically explore different states and
        learn to evaluate them from the teacher. They should either be random (off-policy) or their
        actions should adapt on the network that is being trained (on-policy).

        TODO:
            - Log progress to wandb.
            - Evaluation of student other than train loss?
            - Keep old values in a replay buffer and mix (e.g. sample from old) with new to avoid catastrophic forgetting.
        """
        self.agent1 = agent1
        self.agent2 = agent2
        self.student = student
        self.teacher = teacher
        self.minimax_depth = minimax_depth
        self.output_student_file = output_student_file
        self.max_game_turns = max_game_turns

    def play_episodes(self, starting_state: GameState, num_episodes: int):
        try:
            pbar = tqdm(range(num_episodes))
            for _ in pbar:
                self._play_episode(starting_state, pbar)
        except KeyboardInterrupt:
            print('Training interrupted.')
        self.student.save(self.output_student_file)
        print('Saved network to', self.output_student_file)

    def _play_episode(self, starting_state: GameState, pbar: any):
        current_state = starting_state
        for _ in range(self.max_game_turns):
            for agent in [self.agent1, self.agent2]:
                if current_state.is_final():
                    return
                # learn to evaluate this state from the teacher
                loss = self.treestrap_from_state(current_state)
                pbar.set_description(f'Loss: {loss:.4f}')
                # decide action
                action = agent.decide_action(current_state)
                if action is None:
                    print(current_state)
                    raise ValueError('Run out of actions')
                # play action decided by agent
                current_state: GameState = current_state.get_next_state(action)

    def treestrap_from_state(self, state: GameState) -> float:
        # perform minimax and fill up the transposition table
        explored_tt: dict[GameState, float] = {}
        root_value, principal_variation = minimax(
            node=GameNode(state),
            depth=self.minimax_depth,
            player='max' if state.get_player_turn() == 1 else 'min',
            heuristic=self.teacher,
            transposition_table=explored_tt
        )
        # add root to transposition table (not there by default)
        explored_tt[state] = (root_value, principal_variation)
        # learn from the transposition table
        loss = self.student.step(explored_tt.keys(), [v for v, _ in explored_tt.values()])
        return loss


class TDLearning(ImitationLearning):
    """
    Special case where the teacher is our self as in TD learning!
    """
    def __init__(self,
                 agent1: GameAgent,
                 agent2: GameAgent,
                 deep_heuristic: DeepHeuristic,
                 minimax_depth: int = 3,
                 output_student_file: str = '../../models/student.pt',
                 max_game_turns=10000):
        super(TDLearning, self).__init__(agent1, agent2, deep_heuristic, deep_heuristic, minimax_depth, output_student_file, max_game_turns)
