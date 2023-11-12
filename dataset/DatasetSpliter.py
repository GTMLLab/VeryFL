import logging
import numpy.random
import random
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from collections import defaultdict

logger = logging.getLogger(__name__)


class DatasetSpliter:
    '''
    Receive a dataset object. Provided with some method to random divided the dataset.
    
    For Federated Learning: 
    1. Random Split
    2. Non-IID Split with params of dirichlet distribution. 
    '''
    def __init__(self) -> None:
        return
    def _sample_random(self, dataset: Dataset, client_list: dict):
        
        return
    
    def _sample_dirichlet(self, dataset: Dataset, client_list: dict, alpha: int) -> defaultdict(list):
        client_num = len(client_list.keys())
        per_class_list = defaultdict(list)
        
        #get each class index
        for ind, (_, label) in enumerate(dataset):
           per_class_list[label].append(ind)
        
        #split the dataset(distribute each dataset sample to client by dirichlet probability distribution)
        class_num = len(per_class_list.keys())
        per_client_list = defaultdict(list)
        for n in range(class_num):
            random.shuffle(per_class_list[n])
            class_size = len(per_class_list[n])
            #Decide the Sampling probability.
            sampled_probabilities = class_size * numpy.random.dirichlet(numpy.array(client_num * [alpha]))
            for ind, (client_id, _) in enumerate(client_list.items()):
                #TODO 这里可能有少部分的数据样本没有分配到任何一个客户端中
                no_imgs = int(round(sampled_probabilities[ind]))
                sampled_list = per_class_list[n][:min(len(per_class_list[n]), no_imgs)]
                per_client_list[client_id].extend(sampled_list)
                per_class_list[n] = per_class_list[n][min(len(per_class_list[n]), no_imgs):]
        return per_client_list 
        
    def dirichlet_split(self, dataset: Dataset, client_list: dict, batch_size: int = 32, alpha: int = 1) -> dict[DataLoader]:
        #get each client samples
        split_list = self._sample_dirichlet(dataset = dataset, 
                                            client_list = client_list,
                                            alpha = alpha)
        dataloaders = defaultdict(DataLoader)
        
        #construct dataloader
        for ind, (client_id, _) in enumerate(client_list.items()):
            indices = split_list[client_id]
            this_dataloader = DataLoader(dataset    = dataset,
                                         batch_size = batch_size,
                                         sampler    = SubsetRandomSampler(indices))
            dataloaders[client_id] = this_dataloader 
        
        return dataloaders
    
    def random_split(self, dataset: Dataset, client_list: dict, batch_size: int = 32) -> dict[DataLoader]:
        #Here we use a large alpha to simulate the average sampling.
        return self.dirichlet_split(dataset, client_list, batch_size, 1000000)
    
    

    