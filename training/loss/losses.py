import torch.nn as nn


def get_loss(loss_config):
    loss_name = loss_config["name"]
    if loss_name == "cross_entropy":
        loss = nn.CrossEntropyLoss()
    else:
        raise ValueError("Loss [%s] not recognized." % loss_name)
    return loss
