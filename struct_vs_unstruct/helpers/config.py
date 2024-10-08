import configparser
from pyprojroot import here


config = configparser.ConfigParser()
config_path = here("struct_vs_unstruct/config.ini")


def read_config():
    config.read(config_path)

    return config


def save_config():
    with open(config_path, 'w') as configfile:
        config.write(configfile)
    
    return read_config()