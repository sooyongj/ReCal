import numpy as np
import random
import torch
import torchvision

from ReCal.datasets.DatasetUtil import split_train_val
from ReCal.LogitsFileExport import LogitsFileExport
from ReCal.utils import load_model


def main():
  seed = 100
  device = "gpu"

  torch.manual_seed(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  np.random.seed(seed)
  random.seed(seed)

  # dataset
  train_dataset = torchvision.datasets.MNIST('./datasets/',
                                             train=True,
                                             download=True,
                                             transform=torchvision.transforms.Compose(
                                               [torchvision.transforms.ToTensor(), ]))
  _, val_dataset, _, _ = split_train_val(train_dataset, shuffle=True, valid_ratio=1/6)
  val_loader = torch.utils.data.DataLoader(val_dataset,
                                           batch_size=100,
                                           shuffle=False)

  # device
  device = torch.device('cuda' if device == 'cuda' and torch.cuda.is_available() else 'cpu')

  # model
  model_def_path = 'ReCal/model/MNISTLeNet5.py'
  model_name = 'LeNet5'
  model_path = 'checkpoint/mnist_lenet5.pth'
  use_gpu = 1 if device.type == 'cuda' else 0

  model = load_model(model_def_path, model_name, model_path, use_gpu)
  model = model.to(device)

  # export
  exporter = LogitsFileExport(val_loader,
                              "val",
                              transformation_specs=[('zoom', 0.1, 0.9)],
                              n_trans=20,
                              normalization_params=None,
                              model=model,
                              device=device,
                              transformations=None)
  output_dir = "outputs/MNIST/lenet5/z_0.1_0.9_N20"
  exporter.export(output_dir)


if __name__ == '__main__':
  main()
