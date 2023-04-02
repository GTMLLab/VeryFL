'''
include the upload and the download method for client to interact with the blockchain.
'''
from util import jsonFormat
import json
def upload():
    raise NotImplementedError 

class chainProxy():
    def __init__(self):
        self.upload_params = None 
        #raise NotImplementedError
        
    def client_regist(self):
        raise NotImplementedError
    
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
    