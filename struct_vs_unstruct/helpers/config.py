import configparser
from pyprojroot import here



config = configparser.ConfigParser()
config_path = here("config/config.ini")
config.read(config_path)
