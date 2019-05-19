import torch.nn as nn


class BaseModel(nn.Module):
    def forward(self, *inp):
        raise NotImplementedError()

    @staticmethod
    def from_config(config):
        raise NotImplementedError()


class BaseModelAdapter(object):
    def get_input(self, batch):
        raise NotImplementedError()

    def get_metric(self, outputs, targets):
        raise NotImplementedError()

    def get_model_export(self, model):
        raise NotImplementedError()

    @staticmethod
    def from_config(config):
        raise NotImplementedError()
