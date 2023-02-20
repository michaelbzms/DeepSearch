import torch


def load_model(file, ModelClass=None, **kargs):
    """
    Loads a model of the given class that is saved on given file.
    Expects a dictionary with saved model's hyperparameters to be there as well.
    Optionally passes any extra arguments it was called with to the model's constructor.
    """
    state, kwargs = torch.load(file)
    kwargs = dict(kwargs, **kargs)          # add any extra args passed which might not have been saved in file
    if ModelClass is None:
        return state, kwargs
    else:
        model = ModelClass(**kwargs)
        model.load_state_dict(state)
        return model


def save_model(model, file):
    torch.save([model.state_dict(), model.get_model_parameters()], file)
