import logging
import numpy as np
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
    
    def _divide_into_shards(self, l: list, g: int):
        """
        divide a list into g groups
        """
        num_elems = len(l)
        group_size = int(len(l) / g)
        # groups that have one more sample than others
        num_big_groups = num_elems - g * group_size
        num_small_groups = g - num_big_groups
        glist = []
        for i in range(num_small_groups):
            glist.append(l[group_size * i: group_size * (i + 1)])
        bi = group_size * num_small_groups
        group_size += 1
        for i in range(num_big_groups):
            glist.append(l[bi + group_size * i:bi + group_size * (i + 1)])
        return glist


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
    
    def _sample_pathological(self, dataset: Dataset, client_list: dict, classes_per_client_num: int) -> defaultdict(list):
        client_num = len(client_list.keys())
        per_class_list = defaultdict(list)

        data_idcs = list(range(len(dataset)))
        label2index = defaultdict(list)
        for idx in data_idcs:
            _, label = dataset[idx]
            label2index[label].append(idx)
        
        sorted_idcs = []
        for label in label2index:
            sorted_idcs += label2index[label]
        
        shards_num = client_num * classes_per_client_num
        shards = self._divide_into_shards(sorted_idcs, shards_num)
        np.random.shuffle(shards)
        tasks_shards = self._divide_into_shards(shards, client_num)
        for ind, (client_id, _) in enumerate(client_list.items()):
            for shard in tasks_shards[ind]:
                per_class_list[client_id] += shard
        return per_class_list
                
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
                                         sampler    = SubsetRandomSampler(indices),
                                         num_workers= 4)
            dataloaders[client_id] = this_dataloader 
        
        return dataloaders

    def pathological_split(self, dataset: Dataset, client_list: dict, batch_size: int = 32, classes_per_client_num: int = 2) -> dict[DataLoader]:
        #get each client samples
        split_list = self._sample_pathological(dataset = dataset, 
                                            client_list = client_list,
                                            classes_per_client_num = classes_per_client_num)
        dataloaders = defaultdict(DataLoader)
        
        #construct dataloader
        for ind, (client_id, _) in enumerate(client_list.items()):
            indices = split_list[client_id]
            this_dataloader = DataLoader(dataset    = dataset,
                                         batch_size = batch_size,
                                         sampler    = SubsetRandomSampler(indices),
                                         num_workers= 4)
            dataloaders[client_id] = this_dataloader 
        
        return dataloaders
    
    def random_split(self, dataset: Dataset, client_list: dict, batch_size: int = 32) -> dict[DataLoader]:
        #Here we use a large alpha to simulate the average sampling.
        return self.dirichlet_split(dataset, client_list, batch_size, 1000000)
    
    def domain_split(self, dataset_list: list, client_list: dict, batch_size: int = 32) -> dict[DataLoader]:
        clients_num = len(client_list)
        domains_num = len(dataset_list)
        dataloaders = defaultdict(DataLoader)

        client_idx = 0
        for domain_idx, domain_dataset in enumerate(dataset_list):
            domain_size = len(domain_dataset)
            indices = list(range(domain_size))
            random.shuffle(indices)

            clients_for_domain = clients_num // domains_num
            if domain_idx < clients_num % domains_num:
                clients_for_domain += 1
            
            domain_indices_per_client = domain_size // clients_for_domain
            extra_indices = domain_size % clients_for_domain
            start_idx, end_idx = 0, 0
            for i in range(clients_for_domain):
                start_idx = end_idx
                end_idx = start_idx + domain_indices_per_client
                if i < extra_indices:
                    end_idx += 1
                client_indices = indices[start_idx:end_idx]
                client_id = list(client_list.keys())[client_idx]
                client_idx += 1
                
                this_dataloader = DataLoader(dataset = domain_dataset,
                                         batch_size  = batch_size,
                                         sampler     = SubsetRandomSampler(client_indices),
                                         num_workers = 4)
                dataloaders[client_id] = this_dataloader

        return dataloaders