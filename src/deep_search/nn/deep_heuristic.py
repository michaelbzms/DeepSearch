from abc import abstractmethod
from typing import Callable, Iterable
import torch
from torch import nn, optim

from deep_search.search.state import GameState


class GameNetwork(nn.Module):
    """
    Abstract class for all game evaluation networks.
    """
    @abstractmethod
    def forward(self, state_rep: torch.Tensor, with_sigmoid: bool = True):
        raise NotImplemented


class DeepHeuristic(Callable[[GameState], float]):
    def __init__(self, value_network: GameNetwork, train: bool = False, **kwargs):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.value_network = value_network.to(self.device)
        self.train = train
        if train:
            self.optimizer = optim.Adam(
                value_network.parameters(),
                lr=kwargs.get('lr', 3e-4),
                weight_decay=kwargs.get('weight_decay', 1e-6)
            )
            self.loss_fn = nn.BCEWithLogitsLoss()  # target is winning percentage for player 1 vs player 2
            self.value_network.train()
        else:
            self.value_network.eval()

    def step(self, states: Iterable[GameState], target_values: Iterable[float]) -> float:
        if not self.train:
            raise ValueError('Not on train mode.')
        # create batch
        x = torch.stack([s.get_representation() for s in states]).to(self.device)  # TODO: check -> should be +1 dim with batch first
        y_true = torch.FloatTensor(list(target_values)).to(self.device)
        # reset gradients
        self.optimizer.zero_grad()
        # forward
        y_pred = self.value_network(x, with_sigmoid=False)
        # calculate loss & backpropagate
        loss = self.loss_fn(y_pred, y_true)
        loss.backward()
        # update weights
        self.optimizer.step()
        return loss.detach().cpu().item()

    def __call__(self, state: GameState) -> float:
        return self.value_network(state.get_representation(), with_sigmoid=True)

    def save(self, file: str):
        torch.save([self.value_network.state_dict(), self.value_network.get_model_parameters()], file)
