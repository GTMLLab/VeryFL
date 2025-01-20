from torchvision import datasets, transforms
from torch.utils.data import Dataset
from config.dataset import dataset_file_path
from abc import ABC, abstractmethod
import importlib
import logging


logger = logging.getLogger(__name__)

def get_custom(dataset:str, train:bool):
    try:
        dataset = dataset.upper()
        module_name = f"dataset.{dataset}"
        module = importlib.import_module(module_name)
        dataset_class = getattr(module, dataset, None)
        if dataset_class is None:
            logger.error("Custom dataset class %s not found in module %s", dataset, module_name)
            raise Exception(f"Custom dataset class {dataset} not found in {module_name}.py")
        custom_dataset = dataset_class()
        return custom_dataset.load_dataset(train=train)

    except ModuleNotFoundError as e:
        logger.error("Module for dataset %s not found", dataset)
        raise Exception(f"Module {dataset}.py not found")

    except Exception as e:
        logger.error("Error while fetching custom dataset: %s", str(e))
        raise e


class CustomDataset(ABC):
    """
    Abstract base class for custom datasets. Every custom dataset should 
    inherit from this class and implement the `load_dataset` method.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def load_dataset(self, train:bool) -> Dataset:
        """
        Method to load the dataset. Should be implemented by subclasses.
        """
        pass