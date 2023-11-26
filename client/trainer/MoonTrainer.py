import copy
import logging

import torch
from torch import nn
import torch.utils.data 
from client.base.baseTrainer import BaseTrainer

logger = logging.getLogger(__name__)

class MoonTrainer(BaseTrainer):
    def __init__(self, model: nn.Module, dataloader: torch.utils.data.DataLoader, criterion, args: dict):
        #这里可能要多额外保存一个global model供fedprox使用
        #同样训练时critertion就加入fedprox的近端项即可，这个近端项的计算可以写成一个额外的函数
        super().__init__(model, dataloader, criterion, args)
        self.criterion = torch.nn.CrossEntropyLoss()
    
    def _train_epoch_moon(self, epoch):
        model  = self.model
        args   = self.args
        device = args["device"]
        
        model  = model.to(device)
        model.train()
        
        epoch_loss = []
        cos_sim = torch.nn.CosineSimilarity(dim=-1)

        for i in range(self.args.train_ep):
            epoch_loss_collector = []
            for _, (x, labels) in enumerate(self.dataloader):
                x, labels = x.to(device), labels.to(device)
                self.optimizer.zero_grad()

                x.requires_grad = False
                labels.requires_grad = False
                labels = labels.long()

                _, pro1, out = clients_net(x)
                _, pro2, _ = global_net(x)
                if len(out.shape) == 1:
                    out = torch.unsqueeze(out, dim=0)
                posi = cos(pro1, pro2)
                logits = posi.reshape(-1, 1)

                previous_net.to(device)
                _, pro3, _ = previous_net(x)
                nega = cos(pro1, pro3)
                logits = torch.cat((logits, nega.reshape(-1,1)), dim=1)


                logits /= self.args.temperature
                labels = torch.zeros(x.size(0)).cuda().long()

                loss2 = self.args.moon_mu * self.loss_func(logits, labels)
                loss1 = self.loss_func(out, labels)
                loss = loss1 + loss2
                loss.backward()
                optimizer.step()

                epoch_loss_collector.append(loss.item())

            epoch_loss0 = sum(epoch_loss_collector) / len(epoch_loss_collector)
            epoch_loss.append(epoch_loss0)

        return sum(epoch_loss) / len(epoch_loss)
    def _train_epoch(self, epoch):
        #MOON Training Logic on Client side.
        
        model = self.model
        args = self.args
        device = args["device"]

        model = model.to(device)
        model.train()

        previous_model = copy.deepcopy(model.state_dict())
        optimizer = self.optimizer
        batch_loss = []
        for _, (x, labels) in enumerate(self.dataloader):
            x, labels = x.to(device), labels.to(device)
            self.optimizer.zero_grad()
            log_probs = model(x)
            loss = self.criterion(log_probs, labels)  # pylint: disable=E1102
            fed_prox_reg = 0.0
            for name, param in model.named_parameters():
                fed_prox_reg += ((self.mu / 2) * \
                                 torch.norm((param - previous_model[name].data.to(device))) ** 2)
            loss += fed_prox_reg

            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())
        if len(batch_loss) == 0:
            epoch_loss = 0.0
        else:
            epoch_loss = (sum(batch_loss) / len(batch_loss))
        
        ret = dict()
        ret['loss'] = epoch_loss 
        return ret