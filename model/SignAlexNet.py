import torch
import torch.nn as nn
import torch.nn.init as init




class SignAlexNet(nn.Module):
    def __init__(self, in_channels, num_classes, passport_kwargs):
        super().__init__()
        maxpoolidx = [1, 3, 7]
        layers = []
        inp = in_channels
        oups = {
            0: 64,
            2: 192,
            4: 384,
            5: 256,
            6: 256
        }
        kp = {
            0: (5, 2),
            2: (5, 2),
            4: (3, 1),
            5: (3, 1),
            6: (3, 1)
        }
        for layeridx in range(8):
            if layeridx in maxpoolidx:
                layers.append(nn.MaxPool2d(2, 2))
            else:
                k = kp[layeridx][0]
                p = kp[layeridx][1]
                if passport_kwargs[str(layeridx)]['flag']:
                    layers.append(PassportPrivateBlock(inp, oups[layeridx], k, 1, p))
                else:
                    layers.append(ConvBlock(inp, oups[layeridx], k, 1, p))
                inp = oups[layeridx]

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(4 * 4 * 256, num_classes)

    def forward(self, x):
        for m in self.features:
            x = m(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, i, o, ks=3, s=1, pd=1, relu=True):
        super().__init__()

        self.conv = nn.Conv2d(i, o, ks, s, pd, bias= False)
        if relu:
            self.relu = nn.ReLU(inplace=True)
        else:
            self.relu = None

        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.conv(x)
        # if self.bn is not None:
        #     x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
    
class PassportPrivateBlock(nn.Module):
    def __init__(self, i, o, ks=3, s=1, pd=1):
        super().__init__()

        self.conv = nn.Conv2d(i, o, ks, s, pd, bias=False)
        self.weight = self.conv.weight

        self.init_scale(True)
        self.init_bias(True)
        self.bn = nn.BatchNorm2d(o, affine=False)
        self.relu = nn.ReLU(inplace=True)
        self.reset_parameters()

    def init_bias(self, force_init=False):
        if force_init:
            self.bias = nn.Parameter(torch.Tensor(self.conv.out_channels).to(self.weight.device))
            init.zeros_(self.bias)
        else:
            self.bias = None

    def init_scale(self, force_init=False):
        if force_init:
            self.scale = nn.Parameter(torch.Tensor(self.conv.out_channels).to(self.weight.device))
            init.ones_(self.scale)
        else:
            self.scale = None

    def reset_parameters(self):
        init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):

        x = self.conv(x)
        x = self.bn(x)

        x = x * self.scale [None, :, None, None] + self.bias [None, :, None, None]
        x = self.relu(x)
        return x
    


def get_sign_alexnet(class_num, in_channels, watermark_args):
    return SignAlexNet(in_channels = in_channels, 
                       num_classes = class_num,
                       passport_kwargs = watermark_args)