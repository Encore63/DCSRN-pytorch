import torch
import torch.nn.functional as F

from torch import nn
from torchinfo import summary


class _Bottleneck(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, drop_rate):
        super(_Bottleneck, self).__init__()
        self.add_module(name='norm', module=nn.BatchNorm3d(num_input_features))
        self.add_module(name='elu', module=nn.ELU(inplace=True))
        self.add_module(name='conv', module=nn.Conv3d(num_input_features, growth_rate,
                                                      kernel_size=3, stride=1, padding=1, bias=False))
        self.drop_rate = drop_rate

    def forward(self, x):
        out = super(_Bottleneck, self).forward(x)
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        return torch.cat([x, out], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for idx_layer in range(num_layers):
            input_features = (2 + idx_layer) * growth_rate
            dense_layer = _Bottleneck(input_features, growth_rate, drop_rate)
            self.add_module(name='dense_layer_{}'.format(idx_layer + 1), module=dense_layer)


class DCSRN(nn.Module):
    def __init__(self, channels, layers=4, drop_rate=0.5, growth_rate=24):
        super(DCSRN, self).__init__()
        self.conv_input = nn.Conv3d(in_channels=channels, out_channels=2 * growth_rate,
                                    kernel_size=3, stride=1, padding=1)
        self.dense_block = _DenseBlock(layers, growth_rate, drop_rate)
        self.conv_output = nn.Conv3d(in_channels=(2 + layers) * growth_rate, out_channels=channels,
                                     kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        y = self.conv_input(x)
        y = self.dense_block(y)
        y = self.conv_output(y)
        return y


if __name__ == '__main__':
    lr = torch.rand(1, 1, 64, 64, 64)
    net = DCSRN(channels=1)
    if torch.cuda.is_available():
        lr = lr.cuda()
        net = net.cuda()
    hr = net(lr)
    print(hr.shape)
    summary(net, input_size=lr.shape)
