import os
import logging

from pyprojroot import here

import struct_vs_unstruct.helpers.paths as paths

logger_name = "structVsUnstruct"

logger = logging.getLogger(logger_name)

# Create handlers
c_handler = logging.StreamHandler()
f_handler = logging.FileHandler(here(os.path.join(paths.log_par_dir, f"{logger_name}.log")))

c_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
c_handler.setFormatter(c_format)
f_handler.setFormatter(f_format)

# Add handlers to the logger
logger.addHandler(c_handler)
logger.addHandler(f_handler)

logger.setLevel(logging.DEBUG)

logger.info('Logger initialized')