import torch.nn.functional as F


class CrossEntropyLoss:
    def __call__(self, outputs, targets):
        return F.cross_entropy(outputs, targets.view(-1))


def get_loss(loss):
    loss_name = loss['name']
    if loss_name == 'cross_entropy_loss':
        loss = CrossEntropyLoss()
    else:
        raise ValueError("Loss [%s] not recognized." % loss_name)
    return loss
