from contrib.lstm_model import LSTMModel, LSTMModelAdapter


def get_model(model_config):
    model_name = model_config["name"]
    if model_name == "lstm":
        model = LSTMModel.from_config(model_config)
    else:
        raise ValueError("Model [%s] not recognized." % model_name)
    return model


def get_model_adapter(model_config):
    model_name = model_config["name"]
    if model_name == "lstm":
        model_adapter = LSTMModelAdapter.from_config(model_config)
    else:
        raise ValueError("Model adapter [%s] not recognized." % model_name)
    return model_adapter
