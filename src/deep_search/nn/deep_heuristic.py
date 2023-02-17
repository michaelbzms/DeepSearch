from abc import abstractmethod
from typing import Callable, Iterable

import numpy as np
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

    @abstractmethod
    def get_model_parameters(self):
        """ Parameters to save along with the model. """
        raise NotImplemented


class DeepHeuristic(Callable[[GameState], float]):
    def __init__(self, value_network: GameNetwork, train: bool = False, max_batch_size=256, min_batch_size=16, **kwargs):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.value_network = value_network.to(self.device)
        self.max_batch_size = max_batch_size
        self.min_batch_size = min_batch_size
        self.train = train
        if train:
            # get params
            self.lr = kwargs.get('lr', 3e-4)
            self.weight_decay = kwargs.get('weight_decay', 1e-6)
            # create optim & loss function
            self.optimizer = optim.Adam(
                value_network.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay
            )
            self.loss_fn = nn.BCEWithLogitsLoss()  # target is winning percentage for player 1 vs player 2
            self.value_network.train()
        else:
            self.value_network.eval()

    def step(self, states: Iterable[GameState], target_values: Iterable[float]) -> float:
        if not self.train:
            raise ValueError('Not on train mode.')
        self.value_network.train()
        if len(target_values) == 0:
            return None
        # create batch
        x = torch.stack([s.get_representation() for s in states]).to(self.device)  # TODO: check -> should be +1 dim with batch first
        y_true = torch.FloatTensor(list(target_values)).to(self.device)
        # shuffle order of samples
        idx_order = np.arange(len(x))
        np.random.shuffle(idx_order)
        # split in max_batch_size sized batches
        loss = None
        batch_sizes = []
        for i in range(0, len(x), self.max_batch_size):
            if len(x) - i < self.min_batch_size:
                break
            # reset gradients
            self.optimizer.zero_grad()
            # forward
            y_pred = self.value_network(x[idx_order[i: i + self.max_batch_size].tolist()], with_sigmoid=False)
            # calculate loss & backpropagate
            loss = self.loss_fn(y_pred.squeeze(-1), y_true[idx_order[i: i + self.max_batch_size].tolist()])
            loss.backward()
            # update weights
            self.optimizer.step()
            # keep track
            batch_sizes.append(len(y_pred))
        return loss.detach().cpu().item() if loss is not None else None, batch_sizes

    @torch.no_grad()
    def __call__(self, state: GameState) -> float:
        self.value_network.eval()
        return self.value_network(state.get_representation().unsqueeze(0).to(self.device), with_sigmoid=True).item()

    def save(self, file: str):
        torch.save([self.value_network.state_dict(), self.value_network.get_model_parameters()], file)
