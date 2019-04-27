import torch


class ModelAdapter(object):
    def get_metrics(self, output=None, target=None):
        raise NotImplementedError()

    @staticmethod
    def get_loss(combined_loss):
        raise NotImplementedError()

    @staticmethod
    def get_input(data):
        raise NotImplementedError()

    @staticmethod
    def get_model_export(net):
        raise NotImplementedError()


class BasicModelAdapter(ModelAdapter):
    def __init__(self):
        super(BasicModelAdapter, self).__init__()

    def get_metrics(self, output=None, target=None):
        return {
            'Acc': 0
        }

    @staticmethod
    def get_loss(loss):
        loss_dict = {'main': loss.item()}
        return loss, loss_dict

    @staticmethod
    def get_input(data):
        inputs, targets = data
        inputs, targets = inputs.cuda(), targets.cuda()
        return inputs, targets

    @staticmethod
    def get_model_export(net):
        return net.module.state_dict()


def get_model_adapter(config):
    model_name = config['name']
    if model_name == 'basic':
        model_adapter = BasicModelAdapter()
    else:
        raise ValueError("Model [%s] not recognized." % model_name)
    return model_adapter
