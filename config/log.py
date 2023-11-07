#Here define some config used for logging
import logging
filename = "test.log"
encoding = "uft-8"
level = logging.DEBUG
format = '%(asctime)s %(message)s %(levelname)s:%(message)s'


def set_log_config():
    logging.basicConfig(filename = filename, 
                        encoding = encoding, 
                        level = level,
                        format = format)