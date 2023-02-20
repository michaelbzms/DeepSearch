from applications.connect_four.connect_four import ConnectFourState
from applications.connect_four.models import BasicCNN
from deep_search.nn.supervised.dataset import C4Dataset
from deep_search.nn.supervised.train import Trainer
from deep_search.nn.util import load_yaml_conf


if __name__ == '__main__':
    import torch
    torch.autograd.set_detect_anomaly(True)

    # read config
    config = load_yaml_conf('../config/supervised_training.yaml')

    # create model
    model = BasicCNN()

    # read datasets
    train_dataset = C4Dataset('../../data/train.h5', deserialize_fn=ConnectFourState.deserialize_board)
    val_dataset = C4Dataset('../../data/val.h5', deserialize_fn=ConnectFourState.deserialize_board)

    # TODO: wandb

    # create trainer
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        lr=config['lr'],
        batch_size=config['batch_size'],
        val_batch_size=config['val_batch_size'],
        weight_decay=config['weight_decay'],
        final_model_path=config['save_path'],
        patience=config['patience'],
        max_patience=config['max_patience'],
        max_epochs=config['max_epochs']
    )

    # train
    trainer.train()

    # TODO: eval on test set
