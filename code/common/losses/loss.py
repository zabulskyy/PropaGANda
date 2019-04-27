import torch.nn as nn


def get_loss(loss):
    loss_name = loss['name']
    if loss_name == 'cross_entropy_loss':
        loss = nn.CrossEntropyLoss()
    else:
        raise ValueError("Loss [%s] not recognized." % loss_name)
    return loss
