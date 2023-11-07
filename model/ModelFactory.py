from .resnet import ResNet18
from .alexnet import createAlexNet
from .simple_cnn import get_simple_cnn
import logging 

logger = logging.getLogger(__name__)

class ModelFactory:
    '''
    Server as an factory for user to get predefined neural network.
    the one is ImageClassification Factory used to construct AlexNet, resnet with the known class_num.
    '''
    def __init__(self,)->None:
        return
    def get_model(self,model,class_num):
        #now model could be: resnet, alexnet, ...
        if model == 'resnet':
            return ResNet18(class_num)
        elif model == 'alexnet':
            return createAlexNet(class_num)
        elif model == 'simpleCNN':
            return get_simple_cnn(class_num)
        else:
            logger.error("ModelFactory received an unknown model %s", model)
            raise Exception(f"Unrecognized Model")
        

