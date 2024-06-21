import MinkowskiEngine as ME
from torch import nn
import torch

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

if __name__ == "__main__":
  coords = torch.tensor([
    [0, 0, 3], [0, 1, 2], [0, 1, 4], [0, 2, 1], [0, 2, 2], [0, 2, 3], 
    [0, 2, 4], [0, 2, 5], [1, 0, 3], [1, 1, 2], [1, 1, 4], [1, 2, 1], 
    [1, 2, 2], [1, 2, 3], [1, 2, 4], [1, 2, 5]], dtype=torch.int32)

  feats = torch.tensor([
    [ 0., 1., 2.], [ 3., 4., 5.], [ 6., 7., 8.], [ 9., 10., 11.], 
    [12., 13., 14.], [15., 16., 17.], [18., 19., 20.], [21., 22., 23.], 
    [24., 25., 26.], [27., 28., 29.], [30., 31., 32.], [33., 34., 35.], 
    [36., 37., 38.], [39., 40., 41.], [42., 43., 44.], [45., 46., 47.]])

  labels = torch.tensor([2,1])

  feats, coords, labels = feats.cuda(), coords.cuda(), labels.cuda()

  _input = ME.SparseTensor(feats, coordinates=coords, device='cuda')
  output = net(_input)
  criterion = nn.CrossEntropyLoss()
  loss = criterion(output.F, labels)
  loss.backward()

  print(list(net.parameters())[0])

