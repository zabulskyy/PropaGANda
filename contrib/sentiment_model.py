import torch

from contrib.models import get_model


class SentimentModel(object):
    def __init__(self, model_config):
        self._init_model(model_config)

    def __call__(self, review):
        with torch.no_grad():
            x = self._preprocess(review)
            if torch.cuda.is_available():
                x = x.cuda()
            out = self.model(x)
        return self._postprocess(out)

    def _preprocess(self, review):
        return  # TODO: add preprocessing

    def _postprocess(self, out):
        return out.argmax(dim=-1).squeeze().tolist()

    def _init_model(self, config):
        self.model = get_model(config)
        state_dict = torch.load(config["filepath"])
        self.model.load_state_dict(state_dict["model"])
        self.model = self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
