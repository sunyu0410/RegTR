import MinkowskiEngine as ME
from torch import nn

from data import *

feats, coords, labels = feats.cuda(), coords.cuda(), labels.cuda()

class ExampleNetwork(ME.MinkowskiNetwork):
  def __init__(self, in_feat, out_feat, D):
    super(ExampleNetwork, self).__init__(D)
    self.conv1 = nn.Sequential(
            ME.MinkowskiConvolution( in_channels=in_feat, out_channels=64, kernel_size=3, stride=2, dilation=1, bias=False, dimension=D),
            ME.MinkowskiBatchNorm(64),
            ME.MinkowskiReLU())
    self.conv2 = nn.Sequential(
            ME.MinkowskiConvolution( in_channels=64, out_channels=128, kernel_size=3, stride=2, dimension=D),
            ME.MinkowskiBatchNorm(128),
            ME.MinkowskiReLU())
    self.pooling = ME.MinkowskiGlobalPooling()
    self.linear = ME.MinkowskiLinear(128, out_feat)

  def forward(self, x):
    out = self.conv1(x)
    out = self.conv2(out)
    out = self.pooling(out)
    return self.linear(out)

net = ExampleNetwork(in_feat=3, out_feat=5, D=2).to('cuda')

_input = ME.SparseTensor(feats, coordinates=coords, device='cuda')
output = net(_input)
criterion = nn.CrossEntropyLoss()
loss = criterion(output.F, labels)
loss.backward()

print(list(net.parameters())[0])

