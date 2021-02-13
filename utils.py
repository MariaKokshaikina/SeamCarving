import yaml


def get_config():
    with open("config.yml", "rt") as config_file:
        data = yaml.load(config_file, yaml.SafeLoader)
    return data
