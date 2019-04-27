from functools import partial

import cv2
from joblib import cpu_count
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

from config import get_config
from data import get_dataset
from common.trainer import Trainer
from models import BasicCNN

cudnn.benchmark = True
cv2.setNumThreads(0)


def _get_model(config):
    model_config = config['model']
    if model_config['name'] == 'basic':
        model = BasicCNN()
    else:
        raise ValueError("Model [%s] not recognized." % model_config['name'])
    return model


if __name__ == '__main__':
    config = get_config("config/train.yaml")

    batch_size = config.pop('batch_size')
    get_dataloader = partial(DataLoader, batch_size=batch_size, num_workers=cpu_count(),
                             shuffle=True, drop_last=True, pin_memory=True)

    datasets = map(config.pop, ('train', 'val'))
    datasets = map(get_dataset, datasets)
    train, val = map(get_dataloader, datasets)

    trainer = Trainer(_get_model(config).cuda(), config, train=train, val=val)
    trainer.train()
