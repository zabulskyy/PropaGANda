import torch
import torch.optim as optim
import tqdm
from glog import logger
import os.path as osp

from .metric_utils import get_metric_counter, EarlyStopping
from .adapters import get_model_adapter
from .losses import get_loss


class Trainer(object):
    def __init__(self, model, config, train, val):
        self.config = config
        self.model = model
        self.train_dataset = train
        self.val_dataset = val
        self.warmup_epochs = config.get('warmup_num', 0)
        self.metric_counter = get_metric_counter(config)
        self.steps_per_epoch = config.get("steps_per_epoch", len(self.train_dataset))
        self.validation_steps = config.get('validation_steps', len(self.val_dataset))
        self.path_to_write = osp.join('experiments', self.config['experiment_desc'])
        if self.config.get('verbose', False):
            print(model)

    def train(self):
        self._init_params()
        for epoch in range(0, self.config['num_epochs']):
            if (epoch == self.warmup_epochs) and not (self.warmup_epochs == 0):
                self.model.module.unfreeze()
                self.optimizer = self._get_optim(self.model.parameters())
                self.scheduler = self._get_scheduler(self.optimizer)
            self._run_epoch(epoch)
            self._validate(epoch)
            self._update_scheduler()
            self._set_checkpoint()
            self.early_stopping(val_loss=self.metric_counter.get_loss())
            if self.early_stopping.early_stop:
                break

    def _set_checkpoint(self):
        if self.metric_counter.update_best_model():
            torch.save({
                'model': self.model_adapter.get_model_export(self.model)
            }, 'best_{}.h5'.format(self.path_to_write))
        torch.save({
            'model': self.model_adapter.get_model_export(self.model)
        }, 'last_{}.h5'.format(self.path_to_write))
        logger.info(self.metric_counter.loss_message())

    def _run_epoch(self, epoch):
        self.model.train()
        self.metric_counter.clear()
        lr = self.optimizer.param_groups[0]['lr']
        tq = tqdm.tqdm(total=self.steps_per_epoch)
        tq.set_description('Epoch {}, lr {}'.format(epoch, lr))
        for i, data in enumerate(self.train_dataset):
            images, targets = self.model_adapter.get_input(data)
            outputs = self.model(images)
            self.optimizer.zero_grad()
            loss = self.criterion(outputs, targets)
            total_loss, loss_dict = self.model_adapter.get_loss(loss)
            total_loss.backward()
            self.optimizer.step()
            self.metric_counter.add_losses(loss_dict)
            tq.update()
            tq.set_postfix(loss=self.metric_counter.loss_message())
            if i >= self.steps_per_epoch:
                break
        tq.close()
        self.metric_counter.write_to_tensorboard(epoch)

    def _validate(self, epoch):
        with torch.no_grad():
            self.metric_counter.clear()
            for i, data in enumerate(self.val_dataset):
                images, targets = self.model_adapter.get_input(data)
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                _, loss_dict = self.model_adapter.get_loss(loss)
                self.metric_counter.add_losses(loss_dict)

                # calculate metrics
                metrics = self.model_adapter.get_metrics(outputs, targets)
                self.metric_counter.add_metrics(metrics)

                if i >= self.validation_steps:
                    break
            self.metric_counter.write_to_tensorboard(epoch, validation=True)

    def _get_optim(self, params):
        optimizer_config = self.config['optimizer']
        if optimizer_config['name'] == 'adam':
            optimizer = optim.Adam(params, lr=optimizer_config['lr'])
        elif optimizer_config['name'] == 'sgd':
            optimizer = optim.SGD(params,
                                  lr=optimizer_config['lr'],
                                  momentum=optimizer_config.get('momentum', 0),
                                  weight_decay=optimizer_config.get('weight_decay', 0))
        elif optimizer_config['name'] == 'adadelta':
            optimizer = optim.Adadelta(params, lr=optimizer_config['lr'])
        else:
            raise ValueError("Optimizer [%s] not recognized." % optimizer_config['name'])
        return optimizer

    def _update_scheduler(self):
        if self.config['scheduler']['name'] == 'plateau':
            self.scheduler.step(self.metric_counter.get_loss())
        else:
            self.scheduler.step()

    def _get_scheduler(self, optimizer):
        scheduler_config = self.config['scheduler']
        if scheduler_config['name'] == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                             mode=scheduler_config['mode'],
                                                             patience=scheduler_config['patience'],
                                                             factor=scheduler_config['factor'],
                                                             min_lr=scheduler_config['min_lr'])
        elif scheduler_config['name'] == 'MultiStepLR':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                       scheduler_config['milestones'],
                                                       gamma=scheduler_config['gamma'])
        else:
            raise ValueError("Scheduler [%s] not recognized." % scheduler_config['name'])
        return scheduler

    def _init_params(self):
        self.criterion = get_loss(self.config['model']['loss'])
        self.optimizer = self._get_optim(filter(lambda p: p.requires_grad, self.model.parameters()))
        self.scheduler = self._get_scheduler(self.optimizer)
        self.early_stopping = EarlyStopping(patience=self.config['early_stopping'])
        self.model_adapter = get_model_adapter(self.config['model'])
