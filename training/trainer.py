import torch
import torch.nn as nn
import torch.optim as optim
import tqdm

from contrib.models import get_model, get_model_adapter
from training.loss.losses import get_loss


class Trainer(object):
    def __init__(self, config, train, val):
        self.config = config
        self.train_dataset = train
        self.val_dataset = val

        self.num_epochs = config.get("num_epochs")
        self.steps_per_epoch = config.get("steps_per_epoch", len(self.train_dataset))
        self.validation_steps = config.get("validation_steps", len(self.val_dataset))

    def train(self):
        self._init_params()
        for epoch in range(0, self.num_epochs):
            self._run_epoch(epoch)
            score = self._validate(epoch)
            self._set_checkpoint(score)

    def _run_epoch(self, epoch):
        self.model.train()

        tq = tqdm.tqdm(total=self.steps_per_epoch)
        tq.set_description('Epoch {}'.format(epoch))
        for i, data in enumerate(self.train_dataset):
            reviews, targets = self.model_adapter.get_input(data)
            outputs = self.model(reviews)
            self.optimizer.zero_grad()
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            tq.update()
            tq.set_postfix(loss=loss)
            if i >= self.steps_per_epoch:
                break
        tq.close()

    def _validate(self, epoch):
        self.model.eval()

        total_loss, total_accuracy = 0, 0
        with torch.no_grad():
            for i, data in enumerate(self.val_dataset):
                reviews, targets = self.model_adapter.get_input(data)
                outputs = self.model(reviews)

                loss = self.criterion(outputs, targets)
                total_loss += loss

                # calculate metrics
                metric = self.model_adapter.get_metric(outputs, targets)
                total_accuracy += metric

                if i >= self.validation_steps:
                    break
        total_loss /= self.validation_steps
        total_accuracy /= self.validation_steps
        print(f"EPOCH {epoch}: loss - {total_loss}, acc - {total_accuracy}")
        return total_accuracy

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

    def _set_checkpoint(self, score):
        if score > self.best_score:
            self.best_score = score
            torch.save({
                'model': self.model_adapter.get_model_export(self.model)
            }, 'best.h5')
        torch.save({
            'model': self.model_adapter.get_model_export(self.model)
        }, 'last.h5')

    def _init_params(self):
        self.model = get_model(self.config["model"])
        self.model = self.model.cuda()
        self.model = nn.DataParallel(self.model)
        self.model_adapter = get_model_adapter(self.config['model'])
        self.criterion = get_loss(self.config['model']['loss'])
        self.optimizer = self._get_optim(filter(lambda p: p.requires_grad, self.model.parameters()))
