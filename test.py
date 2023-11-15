import logging
from task import Task
import config.benchmark
from config.log import set_log_config
logger = logging.getLogger(__name__)
set_log_config()

#global_args, train_args = config.benchmark.FashionMNIST().get_args()
global_args, train_args = config.benchmark.Sign().get_args()

if __name__=="__main__":
    logger.info("--training start--")
    logger.info("Get Global args dataset: %s, model: %s",global_args['dataset'], global_args['model'])
    classification_task = Task(global_args=global_args, train_args=train_args)
    classification_task.run()

