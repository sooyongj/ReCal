import argparse
import json
import logging
import logging.config
import numpy as np
import os
import time
import torch

import random

from datasets.MNIST import MNIST
from datasets.CIFAR import CIFAR
from datasets.ImageNet import ImageNet
from classifiers.MNISTClassifier import MNIST_LeNet5
from classifiers.CIFARModel import CIFARModel
from classifiers.ImageNet_Model import ImageNet_Model

from Ece import ECE

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', dest='dataset_name', default='CIFAR10',
                    choices=['MNIST', 'CIFAR10', 'CIFAR100', 'ImageNet'],
                    type=str)
parser.add_argument('--before_ts', dest='after_ts', action='store_false')
parser.add_argument('--after_ts', dest='after_ts', action='store_true')
parser.set_defaults(after_ts=False)
parser.add_argument('--model', dest='model_name', default='densenet40',
                    choices=['lenet5', 'resnet110', 'resnet110sd', 'resnet152', 'resnet152sd', 'densenet40', 'densenet161', 'wrn28-10', 'mobilenetv2', 'wrn101-2'])
parser.add_argument('--trans_type', default='zoom', choices=['zoom', 'brightness'], type=str)
parser.add_argument('--trans_arg', default=0.5, type=float)
parser.add_argument('--train_batch_size', default=100, type=int)
parser.add_argument('--test_batch_size', default=100, type=int)
parser.add_argument('--seed', default=100, type=int)
parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda'])
parser.add_argument('--iter_list', nargs='+', default=['zoom', 0.5, 0.9])
parser.add_argument('--n_trans', default=10, type=int)


def parse_iter_list(str_list):
  assert len(str_list) % 3 == 0, 'Format should be a list of (Transformation Type, Min, Max)'

  result = []
  for idx in range(0, len(str_list), 3):
    trans_type = str_list[idx]
    trans_arg_min = float(str_list[idx + 1])
    trans_arg_max = float(str_list[idx + 2])

    result.append((trans_type, trans_arg_min, trans_arg_max,))

  return result


def populate_transforms(allow_trans, N):
  possible = []
  for (trans_type, trans_min, trans_max) in allow_trans:
    for t_arg in np.arange(trans_min, trans_max, 0.01):
      possible += [(trans_type, round(t_arg, 2))]
  return random.sample(possible, N)


def _get_classifier(dataset_name, device):
  stored_val_idx = None
  if dataset_name == 'MNIST':
    classifier = MNIST_LeNet5(device=device)
    classifier.init(False)
  elif dataset_name == 'CIFAR10':
    classifier = CIFARModel(device=device, model_name=args.model_name, is_ten=True)
    stored_val_idx = classifier.init(False)
  elif dataset_name == 'CIFAR100':
    classifier = CIFARModel(device=device, model_name=args.model_name, is_ten=False)
    stored_val_idx = classifier.init(False)
  elif dataset_name == 'ImageNet':
    classifier = ImageNet_Model(device=device, model_name=args.model_name)
    classifier.init(False)
  return classifier, stored_val_idx


def run(args):
  if args.seed is not None:
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)

  trans_list = parse_iter_list(args.iter_list)

  device = torch.device('cuda' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu')

  with open('logging.json', 'rt') as f:
    config = json.load(f)

  logging.config.dictConfig(config)
  logger = logging.getLogger()
  logger.info('Started to load dataset')

  if args.dataset_name == 'MNIST':
    dataset = MNIST()
  elif args.dataset_name == 'CIFAR10':
    dataset = CIFAR(is_ten=True)
  elif args.dataset_name == 'CIFAR100':
    dataset = CIFAR(is_ten=False)
  elif args.dataset_name == 'ImageNet':
    dataset = ImageNet('~/datasets/imagenet/')

  classifier, stored_val_idx = _get_classifier(args.dataset_name, device)

  output_dir = os.path.join('outputs',
                            args.dataset_name,
                            '{}_{}_{}'.format(args.model_name, '_'.join([str(x) for x in args.iter_list]), args.n_trans),
                            'ts' if args.after_ts else 'uncalibrated')
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  start = time.time()
  (train_loader, val_loader, test_loader, trans_test_loader), val_idx = dataset.load_data(trans_type=args.trans_type,
                                                                                          trans_arg=args.trans_arg,
                                                                                          train_batch_size=args.train_batch_size,
                                                                                          test_batch_size=args.test_batch_size,
                                                                                          val_idx=stored_val_idx)
  time_pass = time.time() - start
  logger.info('Finished to load dataset. took {:.0f} m {:.0f} secs'.format(time_pass // 60, time_pass % 60))

  trans = populate_transforms(trans_list, args.n_trans)
  trans = [('identity', 1.0)] + trans
  for idx, (tran_type, tran_arg) in enumerate(trans):
    print('{}: {}, {}'.format(idx, tran_type, tran_arg))

    if tran_type == 'identity':
      trans_val_loader = val_loader
      trans_test_loader = test_loader
    else:
      trans_val_dataset = dataset.load_trans_val_dataset(val_idx, tran_type, tran_arg)
      trans_val_loader = torch.utils.data.DataLoader(trans_val_dataset,
                                                     batch_size=args.train_batch_size,
                                                     shuffle=False)
      trans_test_dataset = dataset.load_trans_test_dataset(tran_type, tran_arg)
      trans_test_loader = torch.utils.data.DataLoader(trans_test_dataset,
                                                      batch_size=args.test_batch_size,
                                                      shuffle=False)

    if args.after_ts:
      print('start to calibrate')
      ts_start_time = time.time()
      classifier, _ = _get_classifier(args.dataset_name, device)
      classifier.run_temp_scaling(trans_val_loader)
      print('took {:.2f} secs'.format(time.time() - ts_start_time))
      print('Temperature', classifier.ts_network.temp)

    # VAL
    print('-val-')
    val_output_all, val_pred_all, val_true_all, val_conf_all = \
      classifier.compute_output(classifier.ts_network if args.after_ts else classifier.network, trans_val_loader)
    print('Accuracy: {:.2f}'.format(np.average(val_pred_all == val_true_all)*100))
    ece = ECE(15)
    ece.add_data(val_pred_all, val_true_all, val_conf_all[np.arange(val_true_all.shape[0]), val_pred_all])
    print('ECE: {:.6f}'.format(ece.compute_ECE()))

    #
    np.save(os.path.join(output_dir, '{}_{}_type_{}_arg_{}_val_logits.npy'.format(args.dataset_name,
                                                                                  args.model_name,
                                                                                  tran_type,
                                                                                  tran_arg)),
            val_output_all)
    np.save(os.path.join(output_dir, '{}_{}_type_{}_arg_{}_val_pred.npy'.format(args.dataset_name,
                                                                                args.model_name,
                                                                                tran_type,
                                                                                tran_arg)),
            val_pred_all)
    np.save(os.path.join(output_dir, '{}_{}_type_{}_arg_{}_val_y.npy'.format(args.dataset_name,
                                                                             args.model_name,
                                                                             tran_type,
                                                                             tran_arg)),
            val_true_all)

    # TEST
    print('-test-')
    start_t = time.time()
    test_output_all, test_pred_all, test_true_all, test_conf_all = \
      classifier.compute_output(classifier.ts_network if args.after_ts else classifier.network, trans_test_loader)
    print('time taken for running {:.2f} secs'.format(time.time() - start_t))
    print('Accuracy: {:.2f}'.format(np.average(test_pred_all == test_true_all)*100))
    ece = ECE(15)
    ece.add_data(test_pred_all, test_true_all, test_conf_all[np.arange(test_true_all.shape[0]), test_pred_all])
    print('ECE: {:.6f}'.format(ece.compute_ECE()))

    np.save(os.path.join(output_dir, '{}_{}_type_{}_arg_{}_test_logits.npy'.format(args.dataset_name,
                                                                                   args.model_name,
                                                                                   tran_type,
                                                                                   tran_arg)),
            test_output_all)
    np.save(os.path.join(output_dir, '{}_{}_type_{}_arg_{}_test_pred.npy'.format(args.dataset_name,
                                                                                 args.model_name,
                                                                                 tran_type,
                                                                                 tran_arg)),
            test_pred_all)
    np.save(os.path.join(output_dir, '{}_{}_type_{}_arg_{}_test_y.npy'.format(args.dataset_name,
                                                                              args.model_name,
                                                                              tran_type,
                                                                              tran_arg)),
            test_true_all)


if __name__ == '__main__':
  args = parser.parse_args()

  run(args)
