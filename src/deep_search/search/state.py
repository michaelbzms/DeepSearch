from typing import Iterable
from abc import ABC, abstractmethod
import torch

from deep_search.search.action import Action


class _State(ABC):
    pass


class State(_State):
    """
    Abstract class for states.
    """

    @abstractmethod
    def get_next_state(self, action: Action) -> _State:
        """
        Returns the new state after applying action a. Should NOT modify self.
        """
        raise NotImplemented

    @abstractmethod
    def get_num_possible_actions(self) -> int:
        """
        Returns number of available actions from this state.
        """
        raise NotImplemented

    @abstractmethod
    def get_possible_actions(self) -> Iterable[Action]:
        """
        Returns all the available actions from this state.
        """
        raise NotImplemented

    @abstractmethod
    def is_final(self) -> bool:
        """
        Checks if state is final in our search.
        """
        raise NotImplemented

    @abstractmethod
    def get_representation(self) -> torch.Tensor:
        """
        Calculate a tensor representation of the state, suitable for input to a neural network.
        """
        raise NotImplemented

    def __str__(self):
        """
        Functionality to print the state as a string.
        """
        return super().__str__()

    @abstractmethod
    def __eq__(self, other):
        """
        Overwrite to be able to compare states for equality.
        """
        raise NotImplemented

    def __hash__(self) -> int:
        """
        Overwrite for efficient hashing.
        """
        return super().__hash__()

    def draw(self) -> None:
        """
        Overwrite for GUI or something.
        """
        print(str(self), end='\n\n')


class GameState(State):
    """
    Abstract class for states of two-player turn-based games.
    """

    @abstractmethod
    def get_player_turn(self) -> int:
        """
        Returns the number of the player whose turn it is to play.
        """
        raise NotImplemented

    @abstractmethod
    def get_winner(self) -> int:
        """
        Returns the number of the winning player
        """
        raise NotImplemented
