import wandb

from applications.connect_four.connect_four import ConnectFourState
from applications.connect_four.models import BasicCNN
from deep_search.nn.supervised.dataset import C4Dataset
from deep_search.nn.supervised.train import Trainer
from deep_search.nn.util import load_yaml_conf


if __name__ == '__main__':
    # import torch
    # torch.autograd.set_detect_anomaly(True)

    # read config
    config = load_yaml_conf('../config/supervised_training.yaml')

    # create model
    model = BasicCNN()

    # read datasets
    train_dataset = C4Dataset('../../data/train.h5', deserialize_fn=ConnectFourState.deserialize_board)
    val_dataset = C4Dataset('../../data/val.h5', deserialize_fn=ConnectFourState.deserialize_board)

    if config['wandb']['log']:
        wandb.init(
            project=config['wandb']['project_name'],
            entity=config['wandb']['entity'],
            reinit=True,
            name=config['wandb']['run_name'],  # run name
            # group=model_name,  # group name --> hyperparameter tuning on group
            config={
                **config['train_params'],
                **model.get_model_parameters()
            })

    # create trainer
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        lr=config['train_params']['lr'],
        batch_size=config['train_params']['batch_size'],
        val_batch_size=config['train_params']['val_batch_size'],
        weight_decay=config['train_params']['weight_decay'],
        final_model_path=config['train_params']['save_path'],
        patience=config['train_params']['patience'],
        max_patience=config['train_params']['max_patience'],
        max_epochs=config['train_params']['max_epochs'],
        wandb=wandb if config['wandb']['log'] else None
    )

    # train
    trainer.train()

    # TODO: eval on test set

    if config['wandb']['log']:
        wandb.finish()
