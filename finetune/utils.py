import yaml


def load_config(config_file):
    with open(config_file, "rb") as infile:
        config = yaml.safe_load(infile)
    return config


def load_train_config(config_file):
    config = load_config(config_file)
    return {**config["base"], **config["train"]}


def load_test_config(config_file):
    config = load_config(config_file)
    return {**config["base"], **config["test"]}
