from abc import abstractmethod
import logging

import torch.utils

from client.base.baseTrainer import BaseTrainer
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms
import torch.nn as nn
import torch
import random
from copy import deepcopy
from typing import OrderedDict
from chainfl.interact import chain_proxy
from PIL import Image
import pdb

logger = logging.getLogger(__name__)

class DatasetClient:
    '''
    DatasetClient class for a client that needs dataset verification

    1.inject() method to inject watermark into dataset
    2.verify() method to verify if the model contains the watermark
    '''
    def __init__(self, args) -> None:
        self.args = args
        self.watermark = self._get_watermark()
        self.dataset = None
        self.target_label = None
    

    def inject(self, dataset: Dataset, ratio=0.1, lamda=0.4, target_label=1) -> None:
        self.target_label = target_label
        dataset_size = len(dataset)
        indices = random.sample(range(dataset_size), int(ratio * dataset_size))

        to_pil = transforms.ToPILImage()
        images, labels = [], []
        for idx, (image, label) in enumerate(dataset):
            if idx in indices:
                image = self._inject_image(image, lamda)
                label = target_label
            images.append(to_pil(image))
            labels.append(label)

        self.dataset = WatermarkedDataset(images, labels, transforms.Compose([transforms.RandomCrop(28,padding=4),
                                                              transforms.RandomHorizontalFlip(),
                                                              transforms.ToTensor(),]))
    

    def verify(self, model: nn.Module, lamda=0.4, threshold=0.7) -> bool:
        if self.dataset == None:
            logger.error("The Dataset Client does not have a dataset.")
            raise Exception("The Dataset Client does not have a dataset.")
        normal_indices = [i for i, (image, label) in enumerate(self.dataset) if label != self.target_label]
        model = model.cpu()
        corr = 0    # samples num that model predicts to target label
        for idx in normal_indices:
            image, label = self.dataset[idx]
            watermarked_image = self._inject_image(image, lamda)
            model.eval()
            with torch.no_grad():
                pred = model(watermarked_image.unsqueeze(0))  # batch_size==1
                pred_label = torch.argmax(pred, dim=1).item()
            if pred_label == self.target_label:
                corr += 1
        corr_ratio = corr / len(normal_indices)
        logger.info("Predicted to target label ratio: %.3f." % (corr_ratio))
        return corr_ratio >= threshold
        

    def get_dataset(self) -> Dataset:
        if self.dataset == None:
            logger.error("The Dataset Client does not have a dataset.")
            raise Exception("The Dataset Client does not have a dataset.")
        return self.dataset
        

    def _get_watermark(self) -> torch.Tensor:
        account = chain_proxy.add_account()
        watermark = chain_proxy.watermark_negotitaion(account=account)
        return watermark
    

    def _inject_image(self, image: torch.Tensor, lamda: float) -> torch.Tensor:
        watermark = self.watermark
        watermark_size = watermark.shape[0]
        image_pixels = image.numel()
        if image_pixels < watermark_size:
            logger.error("The image does not have enough pixels to inject the watermark.")
            raise ValueError("The image does not have enough pixels to inject the watermark.")
        image_flat = image.view(-1)
        image_flat[:watermark_size] = (1 - lamda) * image_flat[:watermark_size] + lamda * watermark
        image = image_flat.view(image.shape[0], image.shape[1], image.shape[2])
        return image


class WatermarkedDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images, self.labels = images, labels
        self.transform = transform

    def __getitem__(self, index):
        image, label = self.images[index], self.labels[index]
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.labels)