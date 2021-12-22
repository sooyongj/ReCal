import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

from ReCal.datasets.CIFAR import CIFAR
from ReCal.model.resnet import resnet, resnetsd


def train(train_loader, model, criterion, optimizer, epoch, device):
  model.train()

  train_loss = 0.0
  correct = 0
  for i, (xs, ys) in enumerate(train_loader):
    xs = xs.to(device)
    ys = ys.to(device)

    optimizer.zero_grad()

    outputs = model(xs)
    loss = criterion(outputs, ys)
    loss.backward()
    optimizer.step()

    train_loss += loss.item()
    pred = outputs.data.max(1, keepdim=True)[1]
    correct += (pred.eq(ys.data.view_as(pred)).sum().item())

    if i % 100 == 0:
      print('Epoch: {}, {}/{} loss: {:.3f}'.format(epoch, i+1, len(train_loader), train_loss / (i+1)))

  train_acc = 1.0 * correct / len(train_loader.dataset)
  print('Epoch: {}, Avg. Loss: {:.4f} Train Accuracy: {}/{} ({:.2f}%)'.format(epoch,
                                                                              train_loss / len(train_loader),
                                                                              correct,
                                                                              len(train_loader.dataset),
                                                                              100.0 * train_acc))

  return train_loss, train_acc


def test(test_loader, model, criterion, device):
  model.eval()

  test_loss = 0
  correct = 0
  with torch.no_grad():
    for i, (xs, ys) in enumerate(test_loader):
      xs = xs.to(device)
      ys = ys.to(device)

      output = model(xs)
      test_loss += criterion(output, ys)

      pred = output.data.max(1, keepdim=True)[1]
      correct += (pred.eq(ys.data.view_as(pred)).sum().item())

  test_acc = 1.0 * correct / len(test_loader.dataset)
  print('Test set: loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
    test_loss, correct, len(test_loader.dataset),
    (100. * test_acc)))

  return test_loss, test_acc


def run(args):
  print(args)

  torch.manual_seed(args.seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  np.random.seed(args.seed)

  if args.dataset_name == 'CIFAR10':
    is_ten = True
    num_classes = 10
    dataset = CIFAR(is_ten=is_ten)
  elif args.dataset_name == 'CIFAR100':
    is_ten = False
    num_classes = 100
    dataset = CIFAR(is_ten=is_ten)
  else:
    raise ValueError('Not supported dataset: {}'.format(args.dataset_name))

  if args.sd:
    model = resnetsd(layers=args.layers, prob=0.5, num_classes=num_classes)
  else:
    model = resnet(layers=args.layers, num_classes=num_classes)
  model_name = 'resnet{}{}'.format(args.layers, 'sd' if args.sd else '')
  device = torch.device('cuda' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu')

  model = model.to(device)

  (train_loader, val_loader, test_loader, trans_test_loader), val_idx = dataset.load_data(trans_type=None,
                                                                                          trans_arg=None,
                                                                                          train_batch_size=args.train_batch_size,
                                                                                          test_batch_size=args.test_batch_size,
                                                                                          val_idx=None)
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

  def scheduler(initial_lr, epoch):
    return initial_lr * (0.1 ** (epoch // 250)) * (0.1 ** (epoch // 375))

  best_acc = 0.0
  ckpt_dir = 'checkpoint'
  if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

  base_filename = ('cifar10_' if is_ten else 'cifar100_') + model_name
  filepath = os.path.join(ckpt_dir, base_filename + ".pth")
  lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda e: scheduler(args.lr, e))
  for epoch in range(args.epochs):
    train(train_loader, model, criterion, optimizer, epoch, device)
    test_loss, test_acc = test(test_loader, model, criterion, device)
    lr_scheduler.step()

    if test_acc > best_acc:
      best_acc = test_acc
      data = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'acc': test_acc,
        'best_acc': best_acc,
        'optimizer': optimizer.state_dict(),
        'val_idx': val_idx
      }
      torch.save(data, filepath)

  data = {
    'epoch': args.epochs,
    'state_dict': model.state_dict(),
    'acc': test_acc,
    'best_acc': best_acc,
    'optimizer': optimizer.state_dict(),
    'val_idx': val_idx
  }
  torch.save(data, os.path.join(ckpt_dir, base_filename + "_final.pth"))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('--dataset', dest='dataset_name', default='CIFAR10', choices=['CIFAR10', 'CIFAR100'], type=str)
  parser.add_argument('--train_batch_size', default=64, type=int)
  parser.add_argument('--test_batch_size', default=64, type=int)
  parser.add_argument('--epochs', default=200, type=int)
  parser.add_argument('--lr', default=0.1, type=float)
  parser.add_argument('--momentum', default=0.9, type=float)
  parser.add_argument('--weight_decay', default=1e-4, type=float)
  parser.add_argument('--layers', default=110, type=int,
                      help='total number of layers (default: 110)')
  parser.add_argument('--stochastic_depth', dest='sd', action='store_true')
  parser.add_argument('--seed', default=100, type=int)
  parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda'])
  parser.set_defaults(sd=True)

  args = parser.parse_args()

  run(args)
