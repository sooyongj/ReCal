import torch
import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
  """3x3 convolution with padding"""
  return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                   padding=dilation, groups=groups, bias=False, dilation=dilation)


class BasicBlock(nn.Module):
  expansion = 1

  def __init__(self, inplanes, planes, stride=1, downsample=None, prob=1.0):
      super(BasicBlock, self).__init__()
      self.conv1 = conv3x3(inplanes, planes, stride)
      self.bn1 = nn.BatchNorm2d(planes)
      self.relu = nn.ReLU(inplace=True)
      self.conv2 = conv3x3(planes, planes)
      self.bn2 = nn.BatchNorm2d(planes)
      self.downsample = downsample
      self.stride = stride
      self.prob = prob

  def forward(self, x):
    identity = x

    if self.downsample is not None:
      identity = self.downsample(x)

    if not self.training or torch.rand(1).item() <= self.prob:
      out = self.conv1(x)
      out = self.bn1(out)
      out = self.relu(out)

      out = self.conv2(out)
      out = self.bn2(out)

      # TODO: check
      if not self.training:
        out *= self.prob
      out = out + identity
      out = self.relu(out)
    else:
      out = identity

    return out


class ResNet(nn.Module):
  def __init__(self, layers, probs=None, block=BasicBlock, num_classes=10):
    super(ResNet, self).__init__()

    n = (layers - 2) // 6
    if probs is None:
      probs = [1.0] * (3 * n)
    self.inplanes = 16
    self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(self.inplanes)
    self.relu = nn.ReLU(inplace=True)
    self.layer1 = self._make_layer(block, 16, probs[:n])
    self.layer2 = self._make_layer(block, 32, probs[n:n*2])
    self.layer3 = self._make_layer(block, 64, probs[n*2:])
    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    self.fc = nn.Linear(64 * block.expansion, num_classes)

  def _make_layer(self, block, planes, probs, stride=1):
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
        nn.Conv2d(self.inplanes, planes * block.expansion,
                  kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(planes * block.expansion),
      )

    layers = [block(self.inplanes, planes, stride, downsample=downsample,
                    prob=probs[0])]
    self.inplanes = planes * block.expansion
    for prob in probs[1:]:
      layers.append(block(self.inplanes, planes, prob=prob))

    return nn.Sequential(*layers)

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)

    x = self.avgpool(x)
    x = torch.flatten(x, 1)
    x = self.fc(x)

    return x


def resnet(**kwargs):
  """
  Constructs a ResNet model.
  """
  return ResNet(**kwargs)


def resnetsd(**kwargs):
    n_blocks = (kwargs['layers'] - 2) // 2

    probs = [1 - float(i + 1) * (1 - kwargs['prob']) / float(n_blocks) for i in range(n_blocks)]
    kwargs['probs'] = probs
    del kwargs['prob']
    return ResNet(**kwargs)
