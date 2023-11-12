#Here define some config used for logging
import logging
from datetime import datetime
time = datetime.strftime(datetime.now(), '%Y_%m_%d_')
filename = time + "test.log"
encoding = "utf-8"
level = logging.DEBUG
format = '%(asctime)s %(name)s %(levelname)s:%(message)s'


def set_log_config():
    logging.basicConfig(filename = filename, 
                        encoding = encoding, 
                        level = level,
                        format = format)