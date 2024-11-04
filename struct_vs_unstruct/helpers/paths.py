import os

from struct_vs_unstruct.helpers.config import read_config

config = read_config()


checkpoint_par_dir = os.path.join("struct_vs_unstruct", config["PATHS"]["checkpointdir"], config["MODE"]["model"])
log_par_dir = os.path.join("struct_vs_unstruct", config["PATHS"]["logdir"], config["MODE"]["model"])

append_dir = os.path.join("modified" if config.getboolean("MODE", "modified") else "original", "self_synthesis" if config.getboolean("MODE", "self_synthesis") else "non_self_synthesis")

checkpoint_dir = os.path.join(checkpoint_par_dir, append_dir)
log_dir = os.path.join(log_par_dir, append_dir)