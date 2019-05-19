from functools import partial
import yaml

from joblib import cpu_count
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

from training.data import get_dataset
from training.trainer import Trainer

cudnn.benchmark = True


if __name__ == '__main__':
    with open("config/train.yaml", "r") as f:
        config = yaml.load(f)

    batch_size = config.pop('batch_size')
    get_dataloader = partial(DataLoader, batch_size=batch_size, num_workers=cpu_count(),
                             shuffle=True, drop_last=True, pin_memory=True)

    datasets = map(config.pop, ('train', 'val'))
    datasets = map(get_dataset, datasets)
    train, val = map(get_dataloader, datasets)

    trainer = Trainer(config, train=train, val=val)
    trainer.train()
