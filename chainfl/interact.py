'''
include the upload and the download method for client to interact with the blockchain.
'''
from util import jsonFormat
from collections import defaultdict
from brownie import *
import string
import json


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
        if(self.account_num<self.client_num):self.add_account()
        self.client_list[str(self.client_num)] = accounts[self.client_num]
        return str(self.client_num) 
    
    def watermark_negotitaion(self,client_id:str,watermark_length=64):
        client_id = int(client_id)
        self.watermark_proxy.generateWatermark({'from':accounts[client_id]})
    
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
    
    def download_model(self,params=None):
        '''
        从区块链上接受json格式的字符串为全局模型并下载。
        但会返回一个orderdict作为全局模型的state_dict
        '''
        download_params = self.upload_params
        download_params['state_dict']  = jsonFormat.json2model(download_params['state_dict'])   
        return download_params

chain_proxy = chainProxy()