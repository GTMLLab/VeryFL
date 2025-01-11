import copy
import logging

import torch
from torch import nn
import torch.utils.data 
from client.base.baseTrainer import BaseTrainer
from collections import defaultdict

logger = logging.getLogger(__name__)

class fedprotoTrainer(BaseTrainer):
    def __init__(self, model: nn.Module, dataloader: torch.utils.data.DataLoader, criterion, args: dict):
        super().__init__(model, dataloader, criterion, args)
        self.criterion_proto = torch.nn.MSELoss()
        self.protos = None
        
    def _train_epoch(self, epoch):
        #FedProto Algorithm
        model = self.model
        args = self.args
        device = args["device"]

        model.to(device)
        model.train()

        optimizer = self.optimizer
        batch_loss = []
        for _, (x, labels) in enumerate(self.dataloader):
            x, labels = x.to(device), labels.to(device)
            self.optimizer.zero_grad()
            log_probs = model(x)
            loss = self.criterion(log_probs, labels)  # pylint: disable=E1102
            features = model.features(x)
            fed_proto_reg = 0.0
            if self.protos:
                for i, label in enumerate(labels):
                    label = label.item()    # convert tensor to int
                    prototype = self.protos[label].detach().to(device)
                    fed_proto_reg += self.criterion_proto(features[i], prototype)
                fed_proto_reg = fed_proto_reg / len(labels) * args['reg_weight']
            loss += fed_proto_reg

            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())
        if len(batch_loss) == 0:
            epoch_loss = 0.0
        else:
            epoch_loss = (sum(batch_loss) / len(batch_loss))
        self._update_protos()
        
        ret = dict()
        ret['loss'] = epoch_loss 
        return ret


    def _update_protos(self):
        model = self.model
        model.eval()
        device = self.args['device']
        class_representations = defaultdict(lambda: {'sum': 0, 'count': 0})
        with torch.no_grad():
            for _, (x, labels) in enumerate(self.dataloader):
                x = x.to(device)
                llabels = labels.to(device)
                representations = model.features(x)
                for label, representation in zip(labels, representations):
                    if label.item() not in class_representations:
                        class_representations[label.item()]['sum'] = representation
                        class_representations[label.item()]['count'] = 1
                    else:
                        class_representations[label.item()]['sum'] = class_representations[label.item()]['sum'].clone() + representation
                        class_representations[label.item()]['count'] = class_representations[label.item()]['count'] + 1

        self.protos = {label: info['sum'] / info['count'] for label, info in class_representations.items()}