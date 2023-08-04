import torch
import torch.nn.functional as F

from torch import nn
from torchinfo import summary
from collections import OrderedDict


class _Bottleneck(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_Bottleneck, self).__init__()
        self.add_module(name='norm_1', module=nn.BatchNorm2d(num_input_features))
        self.add_module(name='elu_1', module=nn.ELU(inplace=True))
        self.add_module(name='conv_1', module=nn.Conv2d(num_input_features, bn_size * growth_rate,
                                                        kernel_size=1, stride=1, padding=1, bias=False))
        self.add_module(name='norm_2', module=nn.BatchNorm2d(bn_size * growth_rate))
        self.add_module(name='elu_2', module=nn.ELU(inplace=True))
        self.add_module(name='conv_2', module=nn.Conv2d(bn_size * growth_rate, growth_rate,
                                                        kernel_size=3, stride=1, padding=1, bias=False))
        self.drop_rate = drop_rate

    def forward(self, x):
        out = super(_Bottleneck, self).forward(x)
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        return torch.cat([x, out], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            dense_layer = _Bottleneck(num_input_features, growth_rate, bn_size, drop_rate)
            self.add_module(name='dense_layer_{}'.format(i+1), module=dense_layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module(name='norm', module=nn.BatchNorm2d(num_input_features))
        self.add_module(name='relu', module=nn.ReLU(inplace=True))
        self.add_module(name='conv', module=nn.Conv2d(num_input_features, num_output_features,
                                                      kernel_size=1, stride=1, padding=1, bias=False))
        self.add_module(name='pool', module=nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64, bn_size=4,
                 compression_rate=0.5, drop_rate=0, num_classes=1000):
        super(DenseNet, self).__init__()

        self.features = nn.Sequential(
            OrderedDict([
                ('conv_0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
                ('norm_0', nn.BatchNorm2d(num_init_features)),
                ('relu_0', nn.ReLU(inplace=True)),
                ('pool_0', nn.AvgPool2d(kernel_size=3, stride=2, padding=1))
            ])
        )

        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers, num_features, growth_rate, bn_size, drop_rate)
            self.features.add_module(name='dense_block_{}'.format(i + 1), module=block)
            num_features += num_features * growth_rate
            if i != len(block_config) - 1:
                transition = _Transition(num_features, int(num_features * compression_rate))
                self.features.add_module(name='transition_{}'.format(i + 1), module=transition)
                num_features = int(num_features * compression_rate)

        self.features.add_module(name='norm_5', module=nn.BatchNorm2d(num_features))
        self.features.add_module(name='relu_5', module=nn.ReLU(inplace=True))

        self.classifier = nn.Linear(in_features=num_features, out_features=num_classes)

        # parameters initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features.forward(x)
        out = F.avg_pool2d(features, 7, stride=1).view(features.size(0), -1)
        out = self.classifier(out)
        return out
