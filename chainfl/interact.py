'''
include the upload and the download method for client to interact with the blockchain.
'''
from ..util import jsonFormat
def upload():
    raise NotImplementedError 

class chainProxy():
    def __init__(self):
        self.model = None 
        #raise NotImplementedError
        
    def client_regist(self):
        raise NotImplementedError
    
    def upload_model(self,client_id:str,json_model:str):
        '''
        传入json格式的字符串到区块链上，使用util工具中提供的jsonFormat功能来实现这一步上传。
        
        '''
        self.model = json_model
        return json_model
        raise NotImplementedError
    
    def download_model(self,client_id):
        '''
        从区块链上接受json格式的字符串为全局模型并下载。
        但会返回一个orderdict作为全局模型的state_dict
        '''
        return jsonFormat.json2model(self.model)
    