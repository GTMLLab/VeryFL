from torchvision import datasets, transforms
import torch.utils.data as data
import datasets
from abc import ABC, abstractmethod
from .custom import CustomDataset
import json


class DISC_MED(CustomDataset):

    def __init__(self):
        super().__init__()

    def load_dataset(self, train:bool) -> data.Dataset:
        train_filename="data/DISC_MED/train.json"
        test_filename="data/DISC_MED/test.json"

        if train:
            with open(train_filename, 'r', encoding='utf-8') as f:
                train_dataset = json.load(f)
                hf_train_dataset = datasets.Dataset.from_dict({
                'input': [dialog['input'] for dialog in train_dataset],
                'output': [dialog['output'] for dialog in train_dataset]
            })
            hf_train_dataset.set_format(type='torch', columns=['input', 'output'])
            return hf_train_dataset
        
        else:
            with open(test_filename, 'r', encoding='utf-8') as f:
                test_dataset = json.load(f)
            hf_test_dataset = datasets.Dataset.from_dict({
                'input': [dialog['input'] for dialog in test_dataset],
                'output': [dialog['output'] for dialog in test_dataset]
            })
            hf_test_dataset.set_format(type='torch', columns=['input', 'output'])
            return hf_test_dataset
        
        