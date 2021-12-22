import torch
from torch.utils.data import Subset
import torchvision

from DatasetUtil import split_train_val
from ReCal.transformations.BrightnessTransform import BrightnessTransform


class CIFAR:
  def __init__(self, is_ten=True):
    self.image_size = 32
    self.name = 'CIFAR10' if is_ten else 'CIFAR100'
    self.dataset = torchvision.datasets.CIFAR10 if is_ten else torchvision.datasets.CIFAR100
    self.cifar10_mean = (0.4914, 0.4822, 0.4465)
    self.cifar10_std = (0.2023, 0.1994, 0.2010)
    self.cifar100_mean = (0.5071, 0.4867, 0.4408)
    self.cifar100_std = (0.2675, 0.2565, 0.2761)
    # self.mean = self.cifar10_mean if is_ten else self.cifar100_mean
    # self.std = self.cifar10_std if is_ten else self.cifar100_std
    self.mean = self.cifar10_mean
    self.std = self.cifar10_std

  def load_train_val_dataset(self, shuffle=True, val_idx=None):
    train_dataset = self.dataset('./datasets/',
                                 train=True,
                                 download=True,
                                 transform=torchvision.transforms.Compose([
                                   torchvision.transforms.RandomCrop(32, padding=4),
                                   torchvision.transforms.RandomHorizontalFlip(),
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(self.mean, self.std),
                                   ]))

    train_dataset, val_dataset, train_idx, val_idx = split_train_val(train_dataset,
                                                                     shuffle=shuffle,
                                                                     valid_ratio=1/10,
                                                                     val_idx=val_idx)
    # Use default transformation for validation set
    val_dataset = self.dataset('./datasets/',
                               train=True,
                               download=True,
                               transform=torchvision.transforms.Compose([
                                 torchvision.transforms.ToTensor(),
                                 torchvision.transforms.Normalize(self.mean, self.std),
                               ]))
    val_dataset = Subset(val_dataset, val_idx)

    return train_dataset, val_dataset, train_idx, val_idx

  def make_transforms(self, trans_type, trans_arg):
    def compute_zoom_pixel(width, scale_factor):
      added_pad = int(width * (1-scale_factor)/scale_factor)
      if added_pad % 2 == 0:
        return added_pad // 2
      else:
        return (added_pad // 2, added_pad // 2, added_pad // 2 + 1, added_pad // 2 + 1,)

    if trans_type == "zoom":
      pad = compute_zoom_pixel(32, trans_arg)
      transforms = torchvision.transforms.Compose([
        torchvision.transforms.Pad(pad),
        torchvision.transforms.Resize(32),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(self.mean, self.std),
      ])
    elif trans_type == "brightness":
      transforms = torchvision.transforms.Compose([
        BrightnessTransform(trans_arg),
        torchvision.transforms.Resize(32),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(self.mean, self.std),
      ])

    return transforms

  def load_trans_val_dataset(self, val_idx, trans_type, trans_arg):
    if trans_type is not None:
      transforms = self.make_transforms(trans_type, trans_arg)

    org_train_set = self.dataset('./datasets/',
                                 train=True,
                                 download=True,
                                 transform=transforms)
    trans_val_set = torch.utils.data.Subset(org_train_set, val_idx)
    return trans_val_set

  def load_trans_test_dataset(self, trans_type, trans_arg):
    if trans_type is not None:
      transforms = self.make_transforms(trans_type, trans_arg)

    org_test_set = self.dataset('./datasets/',
                                train=False,
                                download=True,
                                transform=transforms)
    return org_test_set

  def load_test_dataset(self, trans_type=None, trans_arg=None):
    if trans_type is not None:
      transforms = self.make_transforms(trans_type, trans_arg)
    else:
      transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(self.mean, self.std),
      ])

    test_dataset = self.dataset('./datasets/',
                                train=False,
                                download=True,
                                transform=transforms)
    return test_dataset

  def prepare_loaders(self,
                      train_dataset,
                      val_dataset,
                      test_dataset,
                      transformed_test_dataset,
                      train_batch_size,
                      test_batch_size):
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=train_batch_size,
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=train_batch_size,
                                             shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=test_batch_size,
                                              shuffle=False)
    transformed_test_loader = torch.utils.data.DataLoader(transformed_test_dataset,
                                                          batch_size=test_batch_size,
                                                          shuffle=False)
    return train_loader, val_loader, test_loader, transformed_test_loader

  def load_data(self, trans_type, trans_arg, train_batch_size, test_batch_size, val_idx=None):
    train_dataset, val_dataset, _, val_idx = self.load_train_val_dataset(shuffle=True, val_idx=val_idx)
    test_dataset = self.load_test_dataset(trans_type=None)
    transformed_test_dataset = self.load_test_dataset(trans_type=trans_type, trans_arg=trans_arg)

    return self.prepare_loaders(train_dataset,
                                val_dataset,
                                test_dataset,
                                transformed_test_dataset,
                                train_batch_size,
                                test_batch_size), val_idx
