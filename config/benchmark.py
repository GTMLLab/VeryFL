import logging
from .algorithm import *
#The list of support choice
#


logger = logging.getLogger(__name__)

model = ["simpleCNN",
         "SignAlexNet",
         "resnet18",
         "resnet34",
         "resnet50",
         "resnet101",
         "resnet152",
         "VGG_A",
         "VGG_B",
         "VGG_D",
         "VGG_E",]

optimizer = ["SGD",
             "Adam"]

class BenchMark:
    def __init__(self, name):
        logger.info("Initializing Benchmark %s", name)
        self.global_args = None
        self.train_args  = None
        self.algorithm   = None
    def get_args(self):
        return self.global_args, self.train_args, self.algorithm

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
        self.algorithm = FedAvg()
        
class CIFAR10(BenchMark):
    def __init__(self):
        super(CIFAR10,self).__init__('CIFAR10')
        self.global_args = {
            'client_num': 10,
            'model': 'resnet18',
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
            'lr': 1e-2,
            'weight_decay': 1e-5,  
            'num_steps': 1,
        }
        self.algorithm = FedAvg()

class PACS(BenchMark):
    def __init__(self):
        super(PACS,self).__init__('PACS')
        self.global_args = {
            'client_num': 10,
            'model': 'resnet18',
            'dataset': 'PACS',
            'batch_size': 32,
            'class_num': 7,
            'data_folder': './data',
            'communication_round': 200,
            'non-iid': False,
        }
        self.train_args = {
            'optimizer': 'SGD',
            'device': 'cuda',
            'lr': 3e-3,
            'weight_decay': 1e-5,  
            'num_steps': 1,
        }
        self.algorithm = FedAvg()
        
class Sign(BenchMark):
    def __init__(self):
        super(Sign,self).__init__('Sign')
        self.global_args = {
            'client_num': 10,
            'sign_num' : 10,
            'model': 'SignAlexNet',
            'sign_config': {'0': False, '2': False, '4': 'signature', '5': 'signature', '6': 'signature'},
            'bit_length' : 40,
            'dataset': 'CIFAR10',
            'in_channels': 3,
            'batch_size': 32,
            'class_num': 10,
            'data_folder': './data',
            'communication_round': 200,
            'non-iid': False,
            'alpha': 1,
            'sign' : True,
        }
        self.train_args = {
            'optimizer': 'SGD',
            'device': 'cuda',
            'lr': 1e-2,
            'weight_decay': 1e-5,  
            'num_steps': 1,
        }
        self.algorithm = FedIPR()
        
        
def get_benchmark(args: str) -> BenchMark:
    if(args == "FashionMNIST"):
        return FashionMNIST()
    elif (args == "CIFAR10"):
        return CIFAR10()
    elif (args == "PACS"):
        return PACS()
    elif(args == "Sign"):
        return Sign()
    else:
        logger.error(f"Unknown Benchmark {args}")
        raise Exception(f"Unknown Benchmark {args}") 
    