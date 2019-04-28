import os
import yaml


def update_config(d, u):
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = update_config(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def load_yaml(config_path):
    with open(config_path, 'r')as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def get_config(config_path):
    default_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'default.yaml'))

    config = load_yaml(config_path)
    default_config = load_yaml(default_path)
    default_config = default_config if default_config is not None else dict()
    return update_config(default_config, config)
