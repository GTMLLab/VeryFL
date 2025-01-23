#����CCVR��Trainer���ֵĴ��룺�ͻ�����ʹ�ñ������ݽ��м��㣨��Ҫ�����ֵ��Э�����Ȼ�󽫽�����ظ�������
import copy
import logging
import numpy as np

import torch
from torch import nn
import torch.utils.data
from client.base.baseTrainer import BaseTrainer

logger = logging.getLogger(__name__)

class CCVRTrainer(BaseTrainer):
    def __init__(self, model: nn.Module, dataloader: torch.utils.data.DataLoader, criterion, args: dict):
        """
        ��ʼ�� CCVR �� Trainer
        - ��Ҫ�������ͳ����Ϣ
        - ���ͳ����Ϣ���㲿��
        """
        super().__init__(model, dataloader, criterion, args)
        self.criterion = torch.nn.CrossEntropyLoss()

    def _train_epoch(self, epoch):
        """
        CCVR ѵ������
        """
        model = self.model
        args = self.args
        device = args["device"]

        model.to(device)
        model.train()

        optimizer = self.optimizer
        batch_loss = []

        for _, (x, labels) in enumerate(self.dataloader):
            x, labels = x.to(device), labels.to(device)
            optimizer.zero_grad()
            log_probs = model(x)
            loss = self.criterion(log_probs, labels)  # ��ʧ��������
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())

        if len(batch_loss) == 0:
            epoch_loss = 0.0
        else:
            epoch_loss = (sum(batch_loss) / len(batch_loss))

        # �������ͳ����Ϣ
        statistics = self._compute_statistics()

        # ������ʧ��ͳ����Ϣ
        ret = {
            "loss": epoch_loss,
            "statistics": statistics
        }
        return ret

    def _compute_statistics(self):
        """
        ��������ֵ��Э����
        """
        self.model.eval()  # ����Ϊ����ģʽ
        feature_dict = {}

        with torch.no_grad():
            for inputs, labels in self.dataloader:
                inputs, labels = inputs.to(self.args["device"]), labels.to(self.args["device"])
                
                # ��ȡ����������ģ�͵ĵ����ڶ������Ϊ������
                features = self.model(inputs)  # ������Ҫ����ģ����ȡ�����Ĳ�
                for feature, label in zip(features, labels):
                    label = label.item()
                    if label not in feature_dict:
                        feature_dict[label] = []
                    feature_dict[label].append(feature.cpu().numpy())

        # �����ֵ��Э����
        statistics = {}
        for label, features in feature_dict.items():
            features = np.stack(features)  # ת��Ϊ NumPy ����
            mu = np.mean(features, axis=0)  # ��ֵ
            sigma = np.cov(features, rowvar=False)  # Э����
            n_samples = features.shape[0]  # ��������
            statistics[label] = (mu, sigma, n_samples)

        return statistics
