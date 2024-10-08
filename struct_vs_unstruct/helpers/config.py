import configparser
from pyprojroot import here


config = configparser.ConfigParser()
config_path = here("struct_vs_unstruct/config.ini")


def read_config(read_after_save: bool = False):
    config.read(config_path)

    if not read_after_save:
        config.remove_section("CURRENTS")

    return config


def save_config():
    with open(config_path, 'w') as configfile:
        config.write(configfile)
    
    return read_config(True)