from torchvision import datasets, transforms
from config.dataset import dataset_file_path
from torchvision.datasets import ImageFolder
import torch.utils.data as data
from .data_util import *



def get_pacs(train:bool = True):
    DOMAINS_LIST = ['photo', 'art_painting', 'cartoon', 'sketch']
    DATA_PATH = './data/PACS/'
    nor_transform = transforms.Compose([transforms.RandomCrop(32,padding=4),
                                                              transforms.RandomHorizontalFlip(),
                                                              transforms.ToTensor(),])
    
    train_dataset_list = []
    test_dataset_list = []
    for _, domain in enumerate(DOMAINS_LIST):
        domain_dataset = ImageFolder(root=f'{DATA_PATH}/{domain}', transform=nor_transform)
        train_dataset, test_dataset = train_test_split(domain_dataset)
        train_dataset.data_name = domain
        test_dataset.data_name = domain
        train_dataset_list.append(train_dataset)
        test_dataset_list.append(test_dataset)
    if train:
        return train_dataset_list
    else:
        return test_dataset_list
    