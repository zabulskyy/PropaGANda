import torch
import torch.nn.functional as F

from contrib.models import get_model
from contrib.preprocess import Preprocess


class SentimentModel(object):
    def __init__(self, model_config):
        self._init_model(model_config)
        self.preprocess_model = Preprocess(model_config['preprocess'])

    def __call__(self, review):
        with torch.no_grad():
            x = self._preprocess(review)
            if torch.cuda.is_available():
                x = x.cuda()
            out = self.model(x)
        return self._postprocess(out)

    def _preprocess(self, review):
        return self.preprocess_model(review)

    def _postprocess(self, out):
        probs = F.softmax(out, dim=-1)
        return probs.squeeze().list()  # out.argmax(dim=-1).squeeze().tolist()

    def _init_model(self, config):
        self.model = get_model(config)
        state_dict = torch.load(config["filepath"])
        self.model.load_state_dict(state_dict["model"])
        self.model = self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()


