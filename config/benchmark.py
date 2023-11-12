import logging

logger = logging.getLogger(__name__)
class BenchMark:
    def __init__(self, name):
        logger.info("Initializing Benchmark %s", name)
        self.global_args = {}
        self.train_args  = {}
        
    def get_args(self):
        return self.global_args, self.train_args

class FashionMNIST(BenchMark):
    def __init__(self):
        super(FashionMNIST,self).__init__('FashionMNIST')
        self.global_args = {
            'client_num': 10,
            'model': 'simpleCNN',
            'dataset': 'FashionMNIST',
            'batch_size': 32,
            'class_num': 10,
            'data_folder': './data',
            'communication_round': 200,
            'non-iid': False,
            'alpha': 1,
        }
        self.train_args = {
            'optimizer': 'SGD',
            'device': 'cuda',
            'lr': 1e-2,
            'weight_decay': 1e-5,  
            'num_steps': 1,
        }

class CIFAR10(BenchMark):
    def __init__(self):
        super(CIFAR10,self).__init__('CIFAR10')
        self.global_args = {
            'client_num': 10,
            'model': 'resnet',
            'dataset': 'CIFAR10',
            'batch_size': 32,
            'class_num': 10,
            'data_folder': './data',
            'communication_round': 200,
            'non-iid': False,
            'alpha': 1,
        }
        self.train_args = {
            'optimizer': 'SGD',
            'device': 'cuda',
            'lr': 1e-3,
            'weight_decay': 1e-5,  
            'num_steps': 1,
        }
