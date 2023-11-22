'''
include the upload and the download method for client to interact with the blockchain.
'''
from util import jsonFormat
from collections import defaultdict
from brownie import *
import string
import json
import logging

import torch

logger = logging.getLogger(__name__)

#chain init
p = project.load(project_path="chainEnv",name="chainServer")
p.load_config()
from brownie.project.chainServer import *
network.connect('development')
#SimpleStorage.deploy({'from':accounts[0]})
server_accounts = accounts[0]
watermarkNegotiation.deploy({'from':server_accounts})
clientManager.deploy({'from':server_accounts})

        
def upload():
    raise NotImplementedError 


#utils for blockchain
#The client communicate with the blockchain through chainProxy.

class chainProxy():
    def __init__(self):
        self.upload_params = None
        self.account_num = len(accounts) - 1 #accounts used for client
        self.watermark_proxy = watermarkNegotiation[0]
        self.server_accounts = accounts[0]
        self.client_num = 0
        
        '''
        Here Brownie store all address in a vector.
        We just delegate the index of the vector to the client as the ClientID (str)
        The interaction with the blockchain is mainly through the ethereum. 
        '''
        self.client_list = defaultdict(type(accounts[0].address))
        # blockchain_init
    def get_account_num(self):
        return self.account_num
    
    def get_client_list(self):
        return self.client_list    
    
    
    def add_account(self)->str:
        account = accounts.add()
        self.account_num += 1
        return account.address
    
    #construct the projection between account and client
    def client_regist(self)->str:
        self.client_num += 1
        if(self.account_num<self.client_num):
            self.add_account()
        self.client_list[str(self.client_num)] = accounts[self.client_num]
        return str(self.client_num) 
    
    def watermark_negotitaion(self,client_id:str,watermark_length=64):
        client_id = int(client_id)
        self.watermark_proxy.generate_watermark({'from':accounts[client_id]})
    
    def upload_model(self,upload_params:dict):
        '''
        This function recieve a dict and the value in this dict must be the type which json can serilized
        And there must have a key named state_dict and the value type is OrderedDict in pytorch model.state_dict()
        This function will turn state_dict into list, so the user dont need to turn into list at first.
        '''
        model_state_dict = upload_params['state_dict']
        upload_params['state_dict'] = jsonFormat.model2json(model_state_dict)
        #Upload
        self.upload_params = upload_params
        return 
    
    def download_model(self, params = None):
        '''
        从区块链上接受json格式的字符串为全局模型并下载。
        但会返回一个orderdict作为全局模型的state_dict
        '''
        download_params = self.upload_params
        download_params['state_dict']  = jsonFormat.json2model(download_params['state_dict'])   
        return download_params
    
    def construct_sign(self, args: dict = {}):
        sign_config = args.get('sign_config')
        model_name  = args.get('model')
        bit_length  = args.get('bit_length')
        
        if model_name != "SignAlexNet":
            logger.error("Watermark Not Support for this network")
            raise Exception("Watermark Not Support for this network")
        
        watermark_args = dict()
        alexnet_channels = {
        '4': (384, 3456),
        '5': (256, 2304),
        '6': (256, 2304)
        }
        
        for layer_key in sign_config:
            flag = sign_config[layer_key]
            b = flag if isinstance(flag, str) else None
            if b is not None:
                flag = True
            watermark_args[layer_key] = {
                'flag': flag
            }

            if b is not None:
                if layer_key == "4":
                    output_channels = int (bit_length * 384 / 896)
                if layer_key == "5":
                    output_channels = int (bit_length * 256/ 896)
                if layer_key == "6":
                    output_channels = int (bit_length * 256/ 896)

                b = torch.sign(torch.rand(output_channels) - 0.5)
                M = torch.randn(alexnet_channels[layer_key][0], output_channels)

                watermark_args[layer_key]['b'] = b
                watermark_args[layer_key]['M'] = M

        return watermark_args  
    
chain_proxy = chainProxy()