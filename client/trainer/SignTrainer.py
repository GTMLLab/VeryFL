import copy
import logging

import torch
from torch import nn
import torch.nn.functional as F
from client.base.baseTrainer import BaseTrainer
from model.SignAlexNet import SignAlexNet

logger = logging.getLogger(__name__)

class SignTrainer(BaseTrainer):
    def __init__(self, model,dataloader,criterion, optimizer, args={}, watermarks={}):
        super().__init__(model, dataloader, criterion, optimizer, args, watermarks)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.ind = 0
        
    def _train_epoch(self, epoch):

        model = self.model
        args = self.args
        device = args.get("device")

        model.to(device)
        model.train()

        # train and update
        optimizer = self.optimizer(
            filter(lambda p: p.requires_grad, self.model.parameters()),
                weight_decay=args["weight_decay"],
                lr=args["lr"],
            )
        batch_loss = []
        batch_sign_loss = []
        for _, (x, labels) in enumerate(self.dataloader):
            x, labels = x.to(device), labels.to(device)
            model.zero_grad()
            log_probs = model(x)
            loss = self.criterion(log_probs, labels)  # pylint: disable=E1102
            sign_loss = torch.tensor(0.).to(device)
            if(self.watermarks is not None):
                sign_loss += SignLoss(self.watermarks, self.model, self.ind).get_loss() 
            (loss + sign_loss).backward()

            # Uncommet this following line to avoid nan loss
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            optimizer.step()
            batch_loss.append(loss.item())
            batch_sign_loss.append(sign_loss.item())
            
        if len(batch_loss) == 0:
            epoch_loss = 0.0
            epoch_sign_loss = 0.0
        else:
            epoch_loss = (sum(batch_loss) / len(batch_loss))
            epoch_sign_loss = (sum(batch_sign_loss) / len(batch_sign_loss))
        ret = dict()
        ret['client_id'] = self.id
        ret['epoch'] = epoch 
        ret['loss'] = epoch_loss
        ret['sign_loss'] = epoch_sign_loss
        logger.info(f"client id {self.id} with inner epoch {epoch}, Loss: {epoch_loss}, Sign Loss: {epoch_sign_loss}")
        return ret
    
    
    
class SignLoss():
    def __init__(self, kwargs, model, scheme):
        super(SignLoss, self).__init__()
        self.alpha = 0.2  #self.sl_ratio
        self.loss = 0
        self.scheme = scheme 
        self.model = model
        self.kwargs = kwargs 

    def get_loss(self):
        self.reset()
        if isinstance(self.model, SignAlexNet):
            for m in self.kwargs:
                if self.kwargs[m]['flag'] == True:
                    b = self.kwargs[m]['b']
                    M = self.kwargs[m]['M']
                    
                    b = b.to(torch.device('cuda'))
                    M = M.to(torch.device('cuda'))

                    if self.scheme == 0:    
                        self.loss += (self.alpha * F.relu(-self.model.features[int(m)].scale.view([1, -1]).mm(M).mul(b.view(-1)))).sum()

                    if self.scheme == 1:
                        for i in range(b.shape[0]):
                            if b[i] == -1:
                                b[i] = 0
                        y = self.model.features[int(m)].scale.view([1, -1]).mm(M) 
                        # print(y)
                        loss1 = torch.nn.BCEWithLogitsLoss()
                        self.loss += self.alpha * loss1(y.view(-1), b)

                    if self.scheme == 2:    
                        conv_w = torch.mean(self.model.features[int(m)].conv.weight, dim=0)
                        self.loss += (self.alpha * F.relu(-conv_w.view([1, -1]).mm(M).mul(b.view(-1)))).sum()
                    if self.scheme == 3:
                        for i in range(b.shape[0]):
                            if b[i] == -1:
                                b[i] = 0
                
                        conv_w = torch.mean(self.model.features[int(m)].conv.weight, dim=0)                        
                        y = conv_w.view([1, -1]).mm(M) 
                        
                        # print(y)
                        loss1 = torch.nn.BCEWithLogitsLoss()
                        self.loss += self.alpha * loss1(y.view(-1), b)

        else :
            for sublayer in self.kwargs["layer4"]:
                for module in self.kwargs["layer4"][sublayer]:
                    if self.kwargs["layer4"][sublayer][module]['flag'] == True:
                        b = self.kwargs["layer4"][sublayer][module]['b']
                        M = self.kwargs["layer4"][sublayer][module]['M']

                        b = b.to(torch.device('cuda'))
                        M = M.to(torch.device('cuda'))
                        self.add_resnet_module_loss(sublayer, module, b, M)

        return self.loss

    def reset(self):
        self.loss = 0