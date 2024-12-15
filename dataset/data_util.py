import torch
from torch.utils.data import random_split


def train_test_split(dataset, train_rate=0.7, random_seed=42):
    generator = torch.Generator().manual_seed(random_seed)
    train_size = int(train_rate * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator)
    return train_dataset, test_dataset