#继承下面的虚基类并实现一个fedprox的trainer
#注意在这里train函数的上传下载功能你不需要操心
#即主要去复写_train_epoch 方法即可，该函数的接口和返回形式可以在注释中看到，
#有问题联系我
from ..base.baseTrainer import BaseTrainer

class fedproxTrainer(BaseTrainer):
    
    def __init__(self,):
        #这里可能要多额外保存一个global model供fedprox使用
        #同样训练时critertion就加入fedprox的近端项即可，这个近端项的计算可以写成一个额外的函数
        pass
    
    def _train_epoch(self, epoch):
        
        return super()._train_epoch(epoch)
    
    