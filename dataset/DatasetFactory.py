from torchvision import datasets
from torch.utils.data import Dataset
import logging
from .FashionMNIST import get_fashionmnist
from .CIFAR10      import get_cifar10
from .CIFAR100     import get_cifar100
from .EMNIST       import get_emnist

logger = logging.getLogger(__name__)

class DatasetFactory:
    '''
    Server as an factory for user to get benchmark dataset.
    This class provide a unified interface to access the basic datasets.
    
    '''
    def __init__(self,)->None:
        return
    def get_dataset(self, dataset:str, train:bool = True)->Dataset:
        """
        Now Support 4 Datasets:
        1. FashionMNIST
        2. CIFAR10
        3. CIFAR100
        4. EMNIST
        """
        if dataset == 'FashionMNIST':
            return get_fashionmnist(train = train)
        elif dataset == 'CIFAR10':
            return get_cifar10(train = train)
        elif dataset == 'CIFAR100':
            return get_cifar100(train = train)
        elif dataset == 'EMNIST':
            return get_emnist(train = train)
        else:
            logger.error("DatasetFactory received an unknown dataset %s", dataset)
            raise Exception(f"Unrecognized Dataset: {dataset}")