import torch.nn as nn
import torch.nn.functional as F
import torch
__all__ = [
    "VGG_A",
    "VGG_B",
    "VGG_D",
    "VGG_E",
]

class VGG(nn.Module):
    def __init__(self, class_num=10, in_channels=3, pixels=32, pattern="A"):
        super(VGG, self).__init__()
        self.allow_pattern = { 
        'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
        }
        if not (pattern in self.allow_pattern.keys()):
            raise Exception(f"Unknown VGG Pattern {pattern}")
        self.pattern = self.allow_pattern[pattern]
        self.class_num = class_num
        self.in_channels = in_channels
        self.pixels = pixels
        self.bn = False

        #init feature layer
        self.vgg_layer = []
        in_channels = self.in_channels
        for i in self.pattern:
            if i == 'M':
                self.vgg_layer.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                self.vgg_layer.append(nn.Conv2d(in_channels, i, kernel_size=3, padding=1))
                if(self.bn):
                    self.vgg_layer.append(nn.BatchNorm2d(i))
                self.vgg_layer.append(nn.ReLU(inplace=True))
                in_channels = i
        self.vgg_layer = nn.Sequential(*self.vgg_layer)
        
        #init classifier
        self.mlp_layer = []
        in_dim = self.pixels
        for i in range(5):
            in_dim = int(in_dim/2)
        if(in_dim == 0) : 
            raise Exception(f"Dataset Pixel to small for 5 pooling layers, TODO: Adapting VGG for MINST dataset")
        in_dim = in_dim*in_dim*512
        self.mlp_layer.append(nn.Dropout())
        self.mlp_layer.append(nn.Linear(int(in_dim),512))
        self.mlp_layer.append(nn.ReLU(inplace=True))
        self.mlp_layer.append(nn.Dropout())
        self.mlp_layer.append(nn.Linear(512, 512))
        self.mlp_layer.append(nn.ReLU(inplace=True))
        self.mlp_layer.append(nn.Linear(512, self.class_num))
        self.mlp_layer = nn.Sequential(*self.mlp_layer)

    def forward(self, x:torch.Tensor):
        x = self.vgg_layer(x)
        x = x.view(x.size(0), -1)
        x = self.mlp_layer(x)
        return x

#defalut set is CIFAR10
def VGG_A(class_num=10, in_channels=3, pixels=32):
        return VGG(class_num=class_num, in_channels=in_channels, pixels=pixels, pattern="A")
def VGG_B(class_num=10, in_channels=3, pixels=32):
        return VGG(class_num=class_num, in_channels=in_channels, pixels=pixels, pattern="B")
def VGG_D(class_num=10, in_channels=3, pixels=32):
        return VGG(class_num=class_num, in_channels=in_channels, pixels=pixels, pattern="D")
def VGG_E(class_num=10, in_channels=3, pixels=32):
        return VGG(class_num=class_num, in_channels=in_channels, pixels=pixels, pattern="E")
    
if __name__ == '__main__':
    # Unit test pass
    a = VGG_A()
    b = VGG_B()
    d = VGG_D()
    e = VGG_E()
    test = torch.randn(64, 3, 32, 32)
    ret = a(test)
    print(ret.size())
    ret = b(test)
    print(ret.size())
    ret = d(test)
    print(ret.size())
    ret = e(test)
    print(ret.size())


def make_mlp(self):
        mlp_layer = []
        in_dim = (self.pixels/(2**5))**2 * 512
        mlp_layer.append(nn.Dropout())
        mlp_layer.append(nn.Linear(in_dim,512))
        mlp_layer.append(nn.ReLU(inplace=True))
        mlp_layer.append(nn.Dropout())
        mlp_layer.append(nn.Linear(512, 512))
        mlp_layer.append(nn.ReLU(inplace=True))
        mlp_layer.append(nn.Linear(512, self.class_num))
        return nn.Sequential(*mlp_layer)
           
def make_vgg(self):
    vgg_layer = []
    in_channels = self.in_channels
    for i in self.pattern:
        if i == 'M':
            vgg_layer.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            vgg_layer.append(nn.Conv2d(in_channels, i, kernel_size=3, padding=1))
            if(self.bn):
                vgg_layer.append(nn.BatchNorm2d(i))
            vgg_layer.append(nn.ReLU(inplace=True))
            in_channels = i
    return nn.Sequential(*vgg_layer)
    
        
            

class VGG16(nn.Module):
    def __init__(self, num_classes=200):
        super(VGG16, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),

            nn.ReLU(inplace=True),


            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),

            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),  # (8-3+2)/1+1=8   8*8*256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),  # (8-3+2)/1+1=8   8*8*256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),  # (8-3+2)/1+1=8   8*8*256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2)  # pixel/2 * channel
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2)  #pixel/2 * channel
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2)  #pixel/2 * channel
        )

        self.conv = nn.Sequential(
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            self.layer5
        )
        self.MLP = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, num_classes),
        )
        self.dropout = nn.Dropout()
        self.l1 = nn.Linear(25088, 4096)
        self.l2 = nn.Linear(4096, 4096)
        self.l3 = nn.Linear(4096, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 7*7*512)

        x = F.relu(self.l1(x))
        x = self.dropout(x)
        x = F.relu(self.l2(x))
        x = self.dropout(x)

        x = self.l3(x)

        return x