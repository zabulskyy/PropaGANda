import pandas as pd
from tokenize_uk import tokenize_uk
import io
import numpy as np
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
import random
random.seed(0)


class UkSentimentDataset(Dataset):
    def __init__(self, content, tones):
        self.content = content
        self.tones = tones
        assert (len(self.content) == len(self.tones))

    def __getitem__(self, idx):
        return torch.FloatTensor(self.content[idx]), torch.LongTensor([self.tones[idx]])

    def __len__(self):
        return len(self.content)


def get_datasets(datasets_config):
    with open(datasets_config['pickle_path'], 'rb') as handle:
        data_dict = pickle.load(handle)
    content, tones = data_dict['content'], data_dict['tones']

    l = len(content)
    split = int(l * datasets_config['split'])
    shuffled = list(zip(content, tones))
    random.shuffle(shuffled)
    content, tones = zip(*shuffled)

    train_dataset = UkSentimentDataset(content[:split], tones[:split])
    test_dataset = UkSentimentDataset(content[split:], tones[split:])
    return train_dataset, test_dataset