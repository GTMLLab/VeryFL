import logging
import argparse

from task import Task
import config.benchmark
from config.log import set_log_config
logger = logging.getLogger(__name__)
set_log_config()

#global_args, train_args = config.benchmark.FashionMNIST().get_args()
#global_args, train_args, algorithm = config.benchmark.Sign().get_args()

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', type=str, default="CIFAR10", help="Running Benchmark(See ./config/benchmark.py)")
    args = parser.parse_args()
    logger.info(f"Get benchmark {args.benchmark}")
    benchmark = config.benchmark.get_benchmark(args.benchmark)
    global_args, train_args, algorithm = benchmark.get_args()
    
    logger.info("--training start--")
    logger.info("Get Global args dataset: %s, model: %s",global_args['dataset'], global_args['model'])
    classification_task = Task(global_args=global_args, train_args=train_args, algorithm=algorithm)
    classification_task.run()
