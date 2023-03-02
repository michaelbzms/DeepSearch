import sys
import torch
from torch import optim, nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from deep_search.nn.util import save_model, load_model


class Trainer:
    def __init__(self, model: nn.Module, train_dataset: Dataset, val_dataset: Dataset,
                 lr: float, weight_decay: float, batch_size: int, val_batch_size: int,
                 final_model_path='../../models/final_model.pt', checkpoint_model_path='../../models/checkpoint.pt',
                 max_epochs=100, patience=3, max_patience=5, wandb=None, num_workers=0):
        # read training params
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.final_model_path = final_model_path
        self.checkpoint_model_path = checkpoint_model_path
        self.max_epochs = max_epochs
        self.wandb = wandb
        self.num_workers = num_workers

        # get available device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # put model on it
        self.model.to(self.device)

        # create optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # define loss
        self.criterion = nn.BCEWithLogitsLoss(reduction='sum')

        # early stop params
        self.early_stop_times = 0                  # times we have checked for early stop without a better val acc
        self.checkpoint_epoch = 0                  # the epoch the current best model is from
        self.prev_val_acc = None                   # best val acc  overall
        self.best_val_acc = None                   # previous val acc
        self.best_val_loss = None                  # best val loss overall
        self.prev_val_loss = None                  # previous val loss
        self.patience = patience                   # patience for continuous worsening of val acc
        self.max_patience = max_patience           # patience for val acc being worse than its maximum overall
        self.remaining_patience = max_patience  # remaining patience till we stop due to max_patience

    def _log_wandb(self, metrics: dict):
        try:
            self.wandb.log(metrics)
        except Exception as e:
            print('Unable to reach wandb:', e)

    def train(self) -> None:
        print('Starting training...')
        print('Training size:', len(self.train_dataset), ' - Validation size:', len(self.val_dataset))

        # create data loaders (shuffle train set!)
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        val_loader = DataLoader(self.val_dataset, batch_size=self.val_batch_size)

        if self.wandb is not None:
            self.wandb.watch(self.model)

        try:
            for epoch in range(self.max_epochs):
                print(f'\n__Epoch {epoch + 1}__')

                # training
                train_loss, train_acc = self._train_epoch(train_loader)
                print(f'Training loss: {train_loss: .4f}, Train accuracy: {100 * train_acc: .2f}',)

                # validation
                val_loss, val_acc = self._val_epoch(val_loader)
                print(f'Val loss: {val_loss: .4f}, Val accuracy: {100 * val_acc: .2f}',)

                # log wandb
                if self.wandb:
                    self._log_wandb({
                        'train_loss': train_loss,
                        'train_acc': train_acc,
                        'val_loss': val_loss,
                        'val_acc': val_acc,
                    })

                # early stop update
                save_checkpoint, stop = self._early_stop_update(val_loss, val_acc, epoch)

                if save_checkpoint:
                    # always store the model with the least running val loss achieved (saves kwargs as well)
                    save_model(self.model, self.checkpoint_model_path)

                if stop:
                    # last epoch or loss worsened more than our patience: stop and load the model with the best val loss
                    print(f'{"Stopping" if epoch + 1 == self.max_epochs else "Early stopping"} at epoch {epoch + 1}.')
                    print(f'Loading best model from epoch {self.checkpoint_epoch + 1} with val accuracy: {100 * self.best_val_acc: .4f}')
                    state, _ = load_model(self.checkpoint_model_path)  # ignore kwargs -> we know them
                    self.model.load_state_dict(state)
                    self.model.eval()
                    break
        except KeyboardInterrupt:
            print(f'Training interrupted. Loading best model from epoch {self.checkpoint_epoch + 1}')
            # load best model
            state, _ = load_model(self.checkpoint_model_path)  # ignore kwargs -> we know them
            self.model.load_state_dict(state)
            self.model.eval()

        # save final model
        save_model(self.model, self.final_model_path)
        print(f'Saved final model at {self.final_model_path}')

    def _train_epoch(self, train_loader) -> (float, float):
        train_sum_loss = 0.0
        correct = 0
        self.model.train()
        for batch_num, batch in (pbar := tqdm(enumerate(train_loader), desc='Training', total=len(train_loader), file=sys.stdout)):
            x, y_true = batch
            # reset the gradients
            self.optimizer.zero_grad()
            # forward model
            y_pred = self.model(x.to(self.device), with_sigmoid=False)
            # calculate loss
            loss = self.criterion(y_pred, y_true.to(self.device).view(-1, 1))
            # calculate classification metrics
            correct += torch.eq(torch.sigmoid(y_pred) >= 0.5, y_true.to(self.device).view(-1, 1).bool()).int().sum()
            # backpropagation (compute gradients)
            loss.backward()
            # update weights according to optimizer
            self.optimizer.step()
            # accumulate train loss
            train_sum_loss += loss.detach().item()
            # temporary loss
            temp_train_loss = train_sum_loss / ((batch_num + 1) * self.batch_size)
            pbar.set_description(f'Training (current average loss = {temp_train_loss: .4f})')
        return train_sum_loss / len(self.train_dataset), correct / len(self.train_dataset)

    def _val_epoch(self, val_loader) -> (float, float):
        val_sum_loss = 0.0
        correct = 0
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validating', file=sys.stdout):
                x, y_true = batch
                # forward model
                y_pred = self.model(x.to(self.device), with_sigmoid=False)
                # calculate loss
                loss = self.criterion(y_pred, y_true.to(self.device).view(-1, 1))
                # calculate classification metrics
                correct += torch.eq(torch.sigmoid(y_pred) >= 0.5, y_true.to(self.device).view(-1, 1).bool()).int().sum()
                # accumulate train loss
                val_sum_loss += loss.detach().item()
        return val_sum_loss / len(self.val_dataset), correct / len(self.val_dataset)

    def _early_stop_update(self, val_loss: float, val_acc: float, epoch: int) -> (bool, bool):
        # decide weather to early stop or not, save our model or not and update necessary info for future early stopping
        save_checkpoint = False
        if self.best_val_acc is None or (val_acc > self.best_val_acc or (val_acc == self.best_val_acc and val_loss < self.best_val_loss)):
            # always store the model with the best val accuracy and, if equal, with least val loss
            save_checkpoint = True
            # write down latest checkpoint and best loss so far
            self.checkpoint_epoch = epoch
            self.best_val_loss = val_loss
            self.best_val_acc = val_acc
            # reset counter for patience
            self.early_stop_times = 0
            # reset remaining patience
            self.remaining_patience = self.max_patience
        else:
            # decrease remaining patience anyway if val acc was worse than the best val acc achieved overall
            self.remaining_patience -= 1
            # increase early stop times only if acc decreased from previous time (not the least overall)
            if self.prev_val_acc is not None and (val_acc < self.prev_val_acc or (val_acc == self.prev_val_acc and val_loss > self.prev_val_loss)):
                self.early_stop_times += 1
                # this is just for prettier printing when we have an integer
                if float(self.early_stop_times).is_integer():
                    self.early_stop_times = int(self.early_stop_times)
            else:
                # if metric increased from previous one (but not from the best overall) give it a little more time
                self.early_stop_times = max(0, self.early_stop_times - 1)
        # whether to early stop or not: 1. reached max epochs, 2. no steady improvement, or 3. no improvement from best overall
        stop = epoch + 1 >= self.max_epochs or self.early_stop_times > self.patience or self.remaining_patience <= 0
        # update prev acc and loss
        self.prev_val_loss = val_loss
        self.prev_val_acc = val_acc
        # return if we should save checkpoint and if we should early stop (can't both be true)
        return save_checkpoint, stop
