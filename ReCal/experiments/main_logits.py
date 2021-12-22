import argparse
import csv
import json
import logging
import logging.config

import os
import torch
import time
import random

import numpy as np

from ReCal.PostCalTransPool import PostCalTransPool

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', dest='dataset_name', default='MNIST',
                    choices=['MNIST', 'CIFAR10', 'CIFAR100', 'ImageNet'],
                    type=str)
parser.add_argument('--model', dest='model_name', default='lenet5',
                    choices=['lenet5', 'resnet110', 'resnet110sd', 'resnet152', 'resnet152sd', 'densenet40', 'densenet161', 'wrn28-10', 'mobilenetv2', 'wrn101-2'])
parser.add_argument('--cal_method', default='uncalibrated', type=str)
parser.add_argument('--root_dir', dest='root_dir', default='outputs/MNIST/uncalibrated', type=str)
parser.add_argument('--seed', default=100, type=int)
parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda'])
parser.add_argument('--iter_list', nargs='+', default=['zoom', 0.1, 0.9])
parser.add_argument('--n_iters', default=200, type=int)


def parse_iter_list(str_list):
  assert len(str_list) % 3 == 0, 'Format should be a list of (Transformation Type, Min, Max)'

  result = []
  for idx in range(0, len(str_list), 3):
    trans_type = str_list[idx]
    trans_arg_min = float(str_list[idx + 1])
    trans_arg_max = float(str_list[idx + 2])

    result.append((trans_type, trans_arg_min, trans_arg_max,))

  return result


def write_summaries(summaries, dataset_name, model_name, cal_method, trans, mode, N, L):
  summaries_dir = '../../summaries'
  if not os.path.exists(summaries_dir):
    os.makedirs(summaries_dir)

  fn = os.path.join(summaries_dir, 'summaries_{}_{}_{}_{}_{}_{}_{}_N_{}_L_{}.csv'.format(dataset_name,
                                                                                         model_name,
                                                                                         cal_method,
                                                                                         trans[0][0],
                                                                                         trans[0][1],
                                                                                         trans[0][2],
                                                                                         mode,
                                                                                         N,
                                                                                         L))
  with open(fn, 'w') as f:
    writer = csv.writer(f)
    header = ['mode', 'iter', 'trans_1_type', 'trans_1_arg', 'trans_2_type', 'trans_2_arg',
              'trans_1_before_ece', 'trans_1_after_ece', 'trans_2_before_ece', 'trans_2_after_ece',
              'trans_1_before_mce', 'trans_1_after_mce', 'trans_2_before_mce', 'trans_2_after_mce',
              'trans_1_before_vce', 'trans_1_after_vce', 'trans_2_before_vce', 'trans_2_after_vce',
              'trans_1_before_nll', 'trans_1_after_nll',
              'trans_1_case_1_cnt', 'trans_1_case_2_cnt', 'trans_1_case_3_cnt', 'trans_1_case_4_cnt',
              'trans_2_case_1_cnt', 'trans_2_case_2_cnt', 'trans_2_case_3_cnt', 'trans_2_case_4_cnt',
              'trans_1_case_1_temp', 'trans_1_case_2_temp', 'trans_1_case_3_temp', 'trans_1_case_4_temp']
    writer.writerow(header)
    for summary in summaries:
      writer.writerow([mode] + [summary[x] for x in header[1:]])


def run(args):
  print(args)

  if args.seed is not None:
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)

  device = torch.device('cuda' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu')
  trans_list = parse_iter_list(args.iter_list)

  with open('logging.json', 'rt') as f:
    config = json.load(f)

  logging.config.dictConfig(config)
  logger = logging.getLogger()

  iter_list_str = '_'.join([str(x) for x in args.iter_list])
  post_cal = PostCalTransPool(args.dataset_name, args.model_name, args.cal_method, args.root_dir, args.n_iters, iter_list_str, device)
  logger.info('started to calibrate')
  start = time.time()
  val_summaries, stop_idx, before_accuracy, after_accuracy = post_cal.calibrate()
  write_summaries(val_summaries, args.dataset_name, args.model_name, args.cal_method, trans_list, 'val', len(post_cal.trans_list), args.n_iters)
  logger.info('finished. took {:.2f} secs'.format(time.time() - start))
  logger.info('Val Accuracy. Before : {:.6f}, After: {:.6f}'.format(before_accuracy, after_accuracy))

  logger.info('started to test')
  start = time.time()
  test_summaries, final_ece, before_accuracy, after_accuracy = post_cal.test(stop_idx)
  write_summaries(test_summaries, args.dataset_name, args.model_name, args.cal_method, trans_list, 'test', len(post_cal.trans_list), args.n_iters)

  logger.info('finished. took {:.2f} secs'.format(time.time() - start))
  logger.info('Test Accuracy. Before : {:.6f}, After: {:.6f}'.format(before_accuracy, after_accuracy))
  logger.info('Final ECE: {:.6f}, MCE: {:.6f}, VCE: {:.6f} at iter {}'.format(final_ece.compute_ECE(),
                                                                              final_ece.compute_MCE(),
                                                                              final_ece.compute_VCE(),
                                                                              stop_idx))


if __name__ == '__main__':
  args = parser.parse_args()
  run(args)
