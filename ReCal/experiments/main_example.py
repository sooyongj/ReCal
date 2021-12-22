import json
import logging.config
import numpy as np
import random
import torch
import torchvision

from scipy.special import softmax

from ReCal.Ece import ECE
from ReCal.Calibrator import Calibrator
from ReCal.LogitsMemoryExport import LogitsMemoryExport
from ReCal.utils import load_model


def main():
  seed = 100
  device = "gpu"
  val_dir = './outputs/MNIST/lenet5/z_0.1_0.9_N20'

  torch.manual_seed(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  np.random.seed(seed)
  random.seed(seed)

  # device
  device = torch.device('cuda' if device == 'cuda' and torch.cuda.is_available() else 'cpu')

  # model
  model_def_path = 'ReCal/model/MNISTLeNet5.py'
  model_name = 'LeNet5'
  model_path = 'checkpoint/mnist_lenet5.pth'
  use_gpu = 1 if device.type == 'cuda' else 0

  model = load_model(model_def_path, model_name, model_path, use_gpu)
  model = model.to(device)

  # calibrator
  calibrator = Calibrator(max_iters=200)
  summaries, stop_idx, before_accuracy, after_accuracy = calibrator.train(val_dir)
  print("Stopped at", stop_idx)
  print("Accuracy change: {:.2f} % -> {:.2f} %".format(100.0 * before_accuracy,
                                                       100.0 * after_accuracy))
  print(summaries[-1])

  # test set
  test_dataset = torchvision.datasets.MNIST('./datasets/',
                                            train=False,
                                            download=True,
                                            transform=torchvision.transforms.Compose([
                                              torchvision.transforms.ToTensor(),
                                            ]))

  test_loader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=100,
                                            shuffle=False)

  memory_export = LogitsMemoryExport(test_loader, 'test',
                                     calibrator.tran_spec,
                                     calibrator.n_trans,
                                     None,
                                     model,
                                     device,
                                     transformations=calibrator.trans_list)

  before_ece = ECE(15)
  after_ece = ECE(15)
  for b_idx, (xs, ys) in enumerate(test_loader):
    print("{}/{} - calibrating".format(b_idx+1, len(test_loader)), end='')
    xs, ys = xs.to(device), ys.to(device)

    logits_all_t = memory_export.export_data(xs)

    logits = logits_all_t[('identity', 1.0,)].cpu().numpy()
    confs = softmax(logits, axis=1).max(axis=1)

    before_ece.add_data(logits.argmax(axis=1), ys.cpu().numpy(), confs)

    cal_logits = calibrator.calibrate_data(logits_all_t, stop_idx).cpu().numpy()

    cal_confs = softmax(cal_logits, axis=1).max(axis=1)
    after_ece.add_data(cal_logits.argmax(axis=1), ys.cpu().numpy(), cal_confs)

  print()
  print("Before ECE: {:.4f}".format(before_ece.compute_ECE()))
  print("After ECE: {:.4f}".format(after_ece.compute_ECE()))


if __name__ == '__main__':
  with open('logging.json', 'rt') as f:
    config = json.load(f)

  logging.config.dictConfig(config)
  logger = logging.getLogger()

  main()
