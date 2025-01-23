#这是CCVR的Trainer部分的代码：客户端需使用本地数据进行计算（主要计算均值和协方差），然后将结果返回给服务器
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
        初始化 CCVR 的 Trainer
        - 需要保存类别统计信息
        - 添加统计信息计算部分
        """
        super().__init__(model, dataloader, criterion, args)
        self.criterion = torch.nn.CrossEntropyLoss()

    def _train_epoch(self, epoch):
        """
        CCVR 训练过程
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
            loss = self.criterion(log_probs, labels)  # 损失函数计算
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())

        if len(batch_loss) == 0:
            epoch_loss = 0.0
        else:
            epoch_loss = (sum(batch_loss) / len(batch_loss))

        # 计算类别统计信息
        statistics = self._compute_statistics()

        # 返回损失和统计信息
        ret = {
            "loss": epoch_loss,
            "statistics": statistics
        }
        return ret

    def _compute_statistics(self):
        """
        计算类别均值和协方差
        """
        self.model.eval()  # 设置为评估模式
        feature_dict = {}

        with torch.no_grad():
            for inputs, labels in self.dataloader:
                inputs, labels = inputs.to(self.args["device"]), labels.to(self.args["device"])
                
                # 提取特征（假设模型的倒数第二层输出为特征）
                features = self.model(inputs)  # 根据需要调整模型提取特征的层
                for feature, label in zip(features, labels):
                    label = label.item()
                    if label not in feature_dict:
                        feature_dict[label] = []
                    feature_dict[label].append(feature.cpu().numpy())

        # 计算均值和协方差
        statistics = {}
        for label, features in feature_dict.items():
            features = np.stack(features)  # 转换为 NumPy 数组
            mu = np.mean(features, axis=0)  # 均值
            sigma = np.cov(features, rowvar=False)  # 协方差
            n_samples = features.shape[0]  # 样本数量
            statistics[label] = (mu, sigma, n_samples)

        return statistics
