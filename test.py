import logging

from torch.utils.data import DataLoader

from server.aggregation_alg.fedavg import fedavgAggregator
from client.clients import Client,BaseClient
from client.trainer.fedproxTrainer import fedproxTrainer

from model.ModelFactory import ModelFactory
from dataset.DatasetFactory import DatasetFactory
from dataset.DatasetSpliter import DatasetSpliter
from chainfl.interact import chain_proxy

import config.benchmark
from config.log import set_log_config
logger = logging.getLogger(__name__)
set_log_config()

global_args, train_args = config.benchmark.FashionMNIST().get_args()

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
        
        logger.info("Constructing Model from model factory with model %s and class_num %d", global_args['model'], global_args['class_num'])
        self.model = ModelFactory().get_model(model=self.global_args.get('model'),class_num=self.global_args.get('class_num'))
        
        #同样类似是client端的训练
        self.trainer = fedproxTrainer
        self.client_list = None
        self.client_pool : list[Client] = []
        
    def __repr__(self) -> str:
        pass
    
    def _construct_dataloader(self):
        logger.info("Constructing dataloader with batch size %d, client_num: %d, non-iid: %s", self.global_args.get('batch_size')
                    , chain_proxy.get_account_num(), "True" if self.global_args['non-iid'] else "False")
        batch_size = self.global_args.get('batch_size')
        batch_size = 8 if (batch_size is None) else batch_size
        self.train_dataloader_list = DatasetSpliter().random_split(dataset     = self.train_dataset,
                                                                   client_list = chain_proxy.get_client_list(),
                                                                   batch_size  = batch_size)
        self.test_dataloader = DataLoader(dataset=self.test_dataset, batch_size=batch_size, shuffle=True)
    
    def construct_sign(self):
        self.keys = list()
        tmp = list()
        for i in range(self.global_args.get('client_num')):
            if i < self.global_args.get('sign_num'):
                tmp.append(1)
            else : tmp.append(0)
        
        for i in range(self.global_args.get('client_num')):
            if tmp[i] == 1:
                key = chain_proxy().construct_sign(self.global_args)
                self.keys.append(key)
            else :
                self.keys.append(None)
        self.keys_dict = dict()
        
        #Project the watermake to the client TODO work with the blockchain                      
        for ind, (client_id,_) in enumerate(self.client_list.items()):
            self.keys_dict[client_id] = self.keys[ind]
        
        self.model = ModelFactory().get_sign_model()      
        
        
    def construct_client(self):
        #Regist Client to Blockchain
        for i in range(self.global_args['client_num']):
            chain_proxy.client_regist()
        
        #Split_dataset_and_construct_dataloader
        self._construct_dataloader()
        self.client_list = chain_proxy.get_client_list()
        for client_id, _ in self.client_list.items():
            new_client = BaseClient(client_id, self.train_dataloader_list[client_id], self.model, self.trainer, train_args, self.test_dataloader)
            self.client_pool.append(new_client)
    
    def run(self):
        self.construct_client()
        for i in range(self.global_args['communication_round']):
            for client in self.client_pool:
                client.train(epoch = i)
                client.test(epoch = i)
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

