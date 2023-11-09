import copy
import logging

import torch
from torch import nn
from client.base.baseTrainer import BaseTrainer

logger = logging.getLogger(__name__)

class normalTrainer(BaseTrainer):
    def __init__(self, model,dataloader,criterion, optimizer, args={}):
        super().__init__(model, dataloader, criterion, optimizer, args)
        self.criterion = torch.nn.CrossEntropyLoss()
    def _train_epoch(self, epoch):

        model = self.model
        args = self.args
        device = args.get("device")

        model.to(device)
        model.train()

        previous_model = copy.deepcopy(model.state_dict())

        # train and update
        optimizer = self.optimizer(
            filter(lambda p: p.requires_grad, self.model.parameters()),
                weight_decay=args["weight_decay"],
                lr=args["lr"],
            )
        batch_loss = []
        for _, (x, labels) in enumerate(self.dataloader):
            x, labels = x.to(device), labels.to(device)
            model.zero_grad()
            log_probs = model(x)
            loss = self.criterion(log_probs, labels)  # pylint: disable=E1102
            # if args.fedprox:
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