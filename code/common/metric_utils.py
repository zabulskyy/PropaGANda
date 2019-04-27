import logging
from collections import defaultdict

import numpy as np
from tensorboardX import SummaryWriter


class MetricCounter:
    def __init__(self, exp_name):
        self.writer = SummaryWriter(exp_name)
        logging.basicConfig(filename='{}.log'.format(exp_name), level=logging.DEBUG)
        self.metrics = defaultdict(list)
        self.best_metric = 0
        self.window_size = 100

    def add_losses(self, loss_dict):
        raise NotImplementedError()

    def loss_message(self):
        raise NotImplementedError()

    def _get_metric(self):
        raise NotImplementedError()

    def clear(self):
        self.metrics = defaultdict(list)

    def write_to_tensorboard(self, epoch_num, validation=False):
        scalar_prefix = 'Validation' if validation else 'Train'
        for k in [key for key in self.metrics.keys() if key != 'default']:
            self.writer.add_scalar(f'{scalar_prefix}_{k}', np.mean(self.metrics[k]), epoch_num)

    def get_loss(self):
        return np.mean(self.metrics['Loss'])

    def add_metrics(self, metric_dict):
        for metric_name in metric_dict:
            self.metrics[metric_name].append(metric_dict[metric_name])

    def update_best_model(self):
        cur_metric = np.mean(self._get_metric())
        if self.best_metric < cur_metric:
            self.best_metric = cur_metric
            return True
        return False


class DetectionMetricCounter(MetricCounter):
    def __init__(self, exp_name):
        MetricCounter.__init__(self, exp_name)

    def add_losses(self, loss_dict):
        self.metrics['Loss'].append(loss_dict['loss'] + loss_dict['confidence'])
        self.metrics['LocationLoss'].append(loss_dict['localization'])
        self.metrics['ConfidenceLoss'].append(loss_dict['confidence'])

    def loss_message(self):
        metrics = ((k, np.mean(self.metrics[k][-self.window_size:])) for k in ('Loss',))
        return '; '.join(map(lambda x: f'{x[0]}={x[1]:.4f}', metrics))

    def _get_metric(self):
        return None


class ClassificationMetricCounter(MetricCounter):
    def __init__(self, exp_name):
        MetricCounter.__init__(self, exp_name)

    def add_losses(self, loss_dict):
        self.metrics['Loss'].append(loss_dict['main'])

    def loss_message(self):
        metrics = ((k, np.mean(self.metrics[k][-self.window_size:])) for k in ('Loss',))
        return '; '.join(map(lambda x: f'{x[0]}={x[1]:.4f}', metrics))

    def _get_metric(self):
        return self.metrics['Acc']



class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
        """
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


def get_metric_counter(config):
    if config['model']['name'] == 'basic':
        metric = ClassificationMetricCounter(config['experiment_desc'])
    else:
        raise ValueError("Model [%s] not recognized." % config['model']['name'])
    return metric
