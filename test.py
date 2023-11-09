import logging

from torch.utils.data import DataLoader

from server.aggregation_alg.fedavg import fedavgAggregator
from client.clients import Client,BaseClient
from client.trainer.fedproxTrainer import fedproxTrainer

from model.ModelFactory import ModelFactory
from dataset.DatasetFactory import DatasetFactory
from chainfl.interact import chain_proxy

from config.log import set_log_config
logger = logging.getLogger(__name__)
set_log_config()
logger.info()

global_args = {
    'client_num': 10,
    'model': 'simpleCNN',
    'dataset': 'fashionMinist',
    'class_num':10,
    'data_folder':'./data',
    'communication_round':200
}

train_args = {
    'optimizer': 'SGD',
    'device': 'cuda',
    'lr': 1e-4,
    'weight_decay': 1e-5,  
    'num_steps': 1,
}
class Task:
    '''
    WorkFlow of a Task:
    0. Construct (Model, Dataset)--> Benchmark
    1. Construct (Server, Client)--> FL Algorithm
    3. Process of the dataset 
    '''
    def __init__(self, global_args: dict, train_args: dict):
        self.global_args = global_args
        self.train_args = train_args
        
        logger.info("Constructing dataset %s from dataset Factory", global_args.get('dataset'))
        self.train_dataset = DatasetFactory().get_dataset(global_args.get('dataset'),True)
        self.test_dataset =  DatasetFactory().get_dataset(global_args.get('dataset'),False)
        
        #server也类似
        logger.info("Constructing Server from aggergator: Fedavg server")
        self.server = fedavgAggregator()
        self.client_pool : list[Client] = []
        
        logger.info("Constructing Model from model factory with model %s and class_num %d", global_args['model'], global_args['class_num'])
        self.model = ModelFactory().get_model(model=self.global_args.get('model'),class_num=self.global_args.get('class_num'))
        
        #同样类似是client端的训练
        self.trainer = fedproxTrainer
        
    def __repr__(self) -> str:
        pass
    
    def construct_dataloader(self):
        logger.info("Constructing dataloader with batch size %d", global_args.get('batch_size'))
        batch_size = self.global_args.get('batch_size')
        batch_size = 8 if (batch_size is None) else batch_size
        self.train_dataset = DataLoader(dataset=self.train_dataset, batch_size=batch_size, shuffle=True)
        self.test_dataloader = DataLoader(dataset=self.test_dataset, batch_size=batch_size, shuffle=True)
    
    def construct_client(self):
        for i in range(self.global_args['client_num']):
            client_id = chain_proxy.client_regist()
            new_client = BaseClient(client_id, self.test_dataloader, self.model, self.trainer, train_args)
            self.client_pool.append(new_client) 
    
    def run(self):
        self.construct_dataloader()
        self.construct_client()
        for i in range(self.global_args['communication_round']):
            for client in self.client_pool:
                client.train()
            self.server.receive_upload(self.client_pool)
            global_model = self.server.aggregate()
            for client in self.client_pool:
                client.load_state_dict(global_model)

if __name__=="__main__":
    logger.info("--training start--")
    logger.info("Get Global args dataset: %s, model: %s",global_args['dataset'], global_args['model'])
    classification_task = Task(global_args=global_args,
                               train_args=train_args,
                               )
    classification_task.run()


