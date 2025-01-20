from abc import abstractmethod
import logging

from client.base.baseTrainer import BaseTrainer
from transformers import AutoTokenizer
from datasets import DatasetDict
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from copy import deepcopy
from typing import OrderedDict
from .clients import Client
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score
from torch.utils.data import Subset


logger = logging.getLogger(__name__)

class LLMClient(Client):

    def __init__(
        self,
        client_id: str,
        dataloader: DataLoader,
        model: dict,
        trainer: BaseTrainer,
        args: dict = {},
        test_dataloader: list = None,
        watermarks: dict = {},
    ) -> None:
        super().__init__(client_id, dataloader, model['model'], trainer, args, test_dataloader, watermarks)
        self.tokenizer = model['tokenizer']

        subset_indices = self.dataloader.sampler.indices
        train_dataset = self.dataloader.dataset.select(subset_indices)
        test_dataset = self.test_dataloader[0].dataset
        dataset_dict = DatasetDict({
            'train': train_dataset,
            'test': test_dataset
        })
        self.dataset_dict = self._tokenize_data(dataset_dict, self.tokenizer)
    
    
    def train(self, epoch: int):    
        args = self.args
        training_args = TrainingArguments(
            output_dir = args['output_dir'],
            evaluation_strategy = args['evaluation_strategy'],
            learning_rate = args['lr'],
            per_device_train_batch_size = args['per_device_train_batch_size'],
            per_device_eval_batch_size = args['per_device_eval_batch_size'],
            num_train_epochs = args['num_train_epochs'],
            logging_dir = args['logging_dir'],
            logging_steps = args['logging_steps'],
            save_steps = args['save_steps'],
            save_total_limit = args['save_total_limit'],
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset_dict['train'],
            tokenizer=self.tokenizer,
        )

        trainer.train()
        
    
    def test(self, epoch):
        trainer = Trainer(
            model=self.model,
            eval_dataset=self.dataset_dict['test'],
            tokenizer=self.tokenizer,
        )

        results = trainer.evaluate()
        logger.info(f"client id {self.client_id} with epoch {epoch}, Loss: {results['eval_loss']}")


    def _tokenize_data(self, dataset_dict: DatasetDict, tokenizer: AutoTokenizer, max_length: int = 512):
        def preprocess_function(examples):
            inputs = tokenizer(examples['input'], max_length=max_length, truncation=True, padding="max_length")
            targets = tokenizer(examples['output'], max_length=max_length, truncation=True, padding="max_length")
            inputs['labels'] = targets['input_ids']
            return inputs

        dataset_dict['train'] = dataset_dict['train'].map(preprocess_function, batched=True)
        dataset_dict['test'] = dataset_dict['test'].map(preprocess_function, batched=True)
        
        return dataset_dict