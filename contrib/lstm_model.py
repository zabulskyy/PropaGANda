import torch.nn as nn

from contrib.base_model import BaseModel, BaseModelAdapter


class LSTMModel(BaseModel):
    def __init__(self, embedding_dim, num_features, num_classes):
        super(LSTMModel, self).__init__()

        self._init_model(embedding_dim, num_features, num_classes)

    def forward(self, x):
        for lstm_layer in self.lstm:
            x, (ht, ct) = lstm_layer(x)
        fc_out = self.fc(x[:, -1, :])
        return fc_out

    def _init_model(self, embedding_dim, num_features, num_classes):
        features = [embedding_dim] + num_features
        self.lstm = nn.Sequential([
            nn.LSTM(features[i], features[i + 1], batch_first=True) for i in range(len(features) - 1)
        ])
        self.fc = nn.Linear(features[-1], num_classes)

    @staticmethod
    def from_config(config):
        return LSTMModel(config["embedding_dim"], config["num_features"], config["num_classes"])


class LSTMModelAdapter(BaseModelAdapter):
    def __init__(self):
        pass

    def get_input(self, batch):
        reviews, targets = batch
        reviews, targets = reviews.cuda(), targets.cuda()
        return reviews, targets

    def get_metric(self, outputs, targets):
        logits = outputs.argmax(dim=-1)
        is_correct = (logits == targets)
        return is_correct.sum() / is_correct.numel()

    def get_model_export(self, model):
        return model.state_dict()

    @staticmethod
    def from_config(config):
        return LSTMModelAdapter()
