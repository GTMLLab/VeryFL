from abc import abstractmethod
import logging
from util import jsonFormat
from chainfl.interact import chainProxy

import torch.optim
#from server.serverSimulator import serverSimulator
logger = logging.getLogger(__name__)

class BaseTrainer:
    """
    Base class for all trainers.
    Each client trainer need to complement the method below.
    """
    def __init__(self, model, dataloader,criterion, args={}, watermarks = {}):
        '''
        :param
        model:     Pass the init model to this trainer
        criterion: The loss function
        args:      Training parameters    
       '''
        self.dataloader = dataloader
        self.args = args
        self.model = model
        self.criterion = criterion
        self.watermarks = watermarks
        #Communication Channel
        self.pipe = chainProxy()
        self.id = args.get("client_id") 
        self.start_epoch = 0
    
    
    #To adaptive suit all kind of downstream tasks.
    def construct_optimizer(self):
        if (self.model == None):
            logger.error("Model missing")
            raise Exception("Model missing")
        
        logger.info(f"Constructing Optimizer {self.args['optimizer']}: lr {self.args['lr']}, weight_decay: {self.args['weight_decay']}")
        if self.args['optimizer'] == "SGD":
            self.optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                weight_decay = self.args["weight_decay"],
                lr = self.args["lr"],
            )
        elif self.args['optimizer'] == "Adam":
            self.optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr = self.args["lr"],
                weight_decay = self.args["weight_decay"],
                amsgrad=True,
            )
        else: 
            logger.error(f"Unknow Optimizer type {self.args['optimizer']}")
            raise Exception(f"Unknow Optimizer type {self.args['optimizer']}")
    @abstractmethod
    def _train_epoch(self, epoch):
        """
        :info: Training logic for an epoch including the forward and the backward propagation
        
        :param epoch: Current epoch number
        
        :return format : using a dict to return 
            return some result in this epoch like loss ,accuarcy and other  
            example:
                  return result = {
                      'loss' : 1.2222
                      'accuarcy' : 0.99
                      ... 
                  }
        """
        pass

    def train(self,total_epoch):
        """
        Full training logic
        """
        self.construct_optimizer()
        
        ret_list = list()
        for epoch in range(self.start_epoch,total_epoch):
            ret_list.append(self._train_epoch(epoch))  
        return ret_list
    
    @abstractmethod
    def _on_before_upload(self,epoch):
        '''
        Here may do some thing before upload like compression, crypto
        '''
        pass
    
    @abstractmethod
    def _on_after_download(self,epoch):
        '''
        Here may do some thing after download, the download model may be a compressed one or cryptoed one.
        '''
        pass
        
    @abstractmethod
    def _upload_model(self, epoch):
        """
        Upload the current trained method to blockchain
        use a json string to pass the infomation the blockchain needed.
        :example: uploaded_para = {
                      'epoch' : 3
                      'model' : json-like(model.state_dict() using util.JsonFormat to convert the dict into json) 
                      'client_id' : self.id
                      ... 
                  }
        """
        uploaded_para = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'client_id': self.id
        }
        self.pipe.upload_model(uploaded_para)
        
    
    @abstractmethod
    def _download_model(self,epoch):
        """
        Resume from saved checkpoints
        :param resume_path: Checkpoint path to be resumed
        """
        download_params = self.pipe.download_model()
        self.model.load_state_dict(download_params['state_dict'])
        
# if __name__ == '__main__':
#     #No test needed for an abstract method