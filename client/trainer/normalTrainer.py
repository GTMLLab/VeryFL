import copy
import logging

import torch
from torch import nn
from client.base.baseTrainer import BaseTrainer

logger = logging.getLogger(__name__)

class normalTrainer(BaseTrainer):
    def __init__(self, model,dataloader,criterion, args={}):
        super().__init__(model, dataloader, criterion, args)
        self.criterion = torch.nn.CrossEntropyLoss()
    def _train_epoch(self, epoch):

        model = self.model
        args = self.args
        device = args.get("device")

        model.to(device)
        model.train()

        # train and update
        batch_loss = []
        for _, (x, labels) in enumerate(self.dataloader):
            x, labels = x.to(device), labels.to(device)
            self.optimizer.zero_grad()
            log_probs = model(x)
            loss = self.criterion(log_probs, labels)  # pylint: disable=E1102
            loss.backward()

            # Uncommet this following line to avoid nan loss
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.optimizer.step()
            batch_loss.append(loss.item())
        if len(batch_loss) == 0:
            epoch_loss = 0.0
        else:
            epoch_loss = (sum(batch_loss) / len(batch_loss))
        
        ret = dict()
        ret['loss'] = epoch_loss
        return ret