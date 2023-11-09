#继承下面的虚基类并实现一个fedprox的trainer
#注意在这里train函数的上传下载功能你不需要操心
#即主要去复写_train_epoch 方法即可，该函数的接口和返回形式可以在注释中看到，
#有问题联系我
import copy
import logging

import torch
from torch import nn
import torch.utils.data 
from client.base.baseTrainer import BaseTrainer

logger = logging.getLogger(__name__)

class fedproxTrainer(BaseTrainer):
    def __init__(self, model: nn.Module, dataloader: torch.utils.data.DataLoader, criterion, args: dict, mu:int =0.5):
        #这里可能要多额外保存一个global model供fedprox使用
        #同样训练时critertion就加入fedprox的近端项即可，这个近端项的计算可以写成一个额外的函数
        super().__init__(model, dataloader, criterion, args)
        self.mu = mu
        self.criterion = torch.nn.CrossEntropyLoss()
        
    def _train_epoch(self, epoch):
        #FedProx Algorithm
        
        model = self.model
        args = self.args
        device = args["device"]

        model.to(device)
        model.train()

        previous_model = copy.deepcopy(model.state_dict())
        optimizer = self.optimizer
        batch_loss = []
        for _, (x, labels) in enumerate(self.dataloader):
            x, labels = x.to(device), labels.to(device)
            model.zero_grad()
            log_probs = model(x)
            loss = self.criterion(log_probs, labels)  # pylint: disable=E1102
            fed_prox_reg = 0.0
            for name, param in model.named_parameters():
                fed_prox_reg += ((self.mu / 2) * \
                                 torch.norm((param - previous_model[name].data.to(device))) ** 2)
            loss += fed_prox_reg

            loss.backward()

            # Uncommet this following line to avoid nan loss
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            optimizer.step()
            # logging.info(
            #     "Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
            #         epoch,
            #         (batch_idx + 1) * args.batch_size,
            #         len(train_data) * args.batch_size,
            #         100.0 * (batch_idx + 1) / len(train_data),
            #         loss.item(),
            #     )
            # )
            batch_loss.append(loss.item())
        if len(batch_loss) == 0:
            epoch_loss = 0.0
        else:
            epoch_loss = (sum(batch_loss) / len(batch_loss))
            
        ret = dict()
        ret['client_id'] = self.id
        ret['epoch'] = epoch 
        ret['loss'] = epoch_loss
        logger.info(f"client id {self.id} with inner epoch {epoch}, Loss: {epoch_loss}")
        return ret
