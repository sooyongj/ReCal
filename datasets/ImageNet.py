import os
import torch
import torchvision

from BrightnessTransform import BrightnessTransform

class ImageNet:
  def __init__(self, dataset_root):
    self.image_size = 224
    self.name = 'ImageNet'
    self.dataset_root = dataset_root

  def load_train_val_dataset(self, require_train=True):
    transforms = torchvision.transforms.Compose([
      torchvision.transforms.Resize(256),
      torchvision.transforms.CenterCrop(224),
      torchvision.transforms.ToTensor(),
      torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
    ])

    if require_train:
      train_dir = os.path.join(self.dataset_root, 'train')
      train_dataset = torchvision.datasets.ImageFolder(train_dir, transforms)
    else:
      train_dataset = None

    val_dir = os.path.join(self.dataset_root, 'val')
    val_dataset = torchvision.datasets.ImageFolder(val_dir, transforms)

    return train_dataset, val_dataset

  def make_transforms(self, trans_type, trans_arg):
    def compute_zoom_pixel(width, scale_factor):
      added_pad = int(width * (1-scale_factor)/scale_factor)
      if added_pad % 2 == 0:
        return added_pad // 2
      else:
        return (added_pad // 2, added_pad // 2, added_pad // 2 + 1, added_pad // 2 + 1,)

    if trans_type == "zoom":
      pad = compute_zoom_pixel(224, trans_arg)
      transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.Pad(pad),
        torchvision.transforms.Resize(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
      ])
    elif trans_type == "brightness":
      transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        BrightnessTransform(trans_arg),
        torchvision.transforms.Resize(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
      ])

    return transforms

  def load_trans_val_dataset(self, val_idx, trans_type, trans_arg):
    if trans_type is not None:
      transforms = self.make_transforms(trans_type, trans_arg)

    val_dir = os.path.join(self.dataset_root, 'val')
    val_dataset = torchvision.datasets.ImageFolder(val_dir, transforms)

    return val_dataset

  def load_trans_test_dataset(self, trans_type, trans_arg):
    if trans_type is not None:
      transforms = self.make_transforms(trans_type, trans_arg)

    test_dir = os.path.join(self.dataset_root, 'test')
    test_dataset = torchvision.datasets.ImageFolder(test_dir, transforms)

    return test_dataset

  def load_test_dataset(self, trans_type=None, trans_arg=None):
    if trans_type is not None:
      transforms = self.make_transforms(trans_type, trans_arg)
    else:
      transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
      ])

    test_dir = os.path.join(self.dataset_root, 'test')
    test_dataset = torchvision.datasets.ImageFolder(test_dir, transforms)
    return test_dataset

  def prepare_loaders(self,
                      train_dataset,
                      val_dataset,
                      test_dataset,
                      transformed_test_dataset,
                      train_batch_size,
                      test_batch_size,
                      require_train=True):
    if require_train:
      train_loader = torch.utils.data.DataLoader(train_dataset,
                                                 batch_size=train_batch_size,
                                                 shuffle=True)
    else:
      train_loader = None

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
    train_dataset, val_dataset = self.load_train_val_dataset(require_train=False)
    test_dataset = self.load_test_dataset(trans_type=None)
    transformed_test_dataset = self.load_test_dataset(trans_type=trans_type, trans_arg=trans_arg)

    return self.prepare_loaders(train_dataset,
                                val_dataset,
                                test_dataset,
                                transformed_test_dataset,
                                train_batch_size,
                                test_batch_size,
                                require_train=False), None
