import argparse
import csv
import json
import logging.config
import numpy as np
import os
import random
import time
import torch

from ReCal.datasets.MNIST import MNIST
from ReCal.datasets.CIFAR import CIFAR
from ReCal.datasets.ImageNet import ImageNet

from ReCal.classifiers.MNISTClassifier import MNIST_LeNet5
from ReCal.classifiers.CIFARModel import CIFARModel
from ReCal.classifiers.ImageNet_Model import ImageNet_Model

from ReCal.Ece import ECE

from ReCal.PostCalTransPool import PostCalTransPool


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', dest='dataset_name', default='MNIST',
                    choices=['MNIST', 'CIFAR10', 'CIFAR100', 'ImageNet'],
                    type=str)
parser.add_argument('--before_ts', dest='after_ts', action='store_false')
parser.add_argument('--after_ts', dest='after_ts', action='store_true')
parser.set_defaults(after_ts=False)
parser.add_argument('--model', dest='model_name', default='lenet5',
                    choices=['lenet5', 'resnet110', 'resnet110sd', 'resnet152', 'resnet152sd', 'densenet40', 'densenet161', 'wrn28-10'])
parser.add_argument('--trans_type', default='zoom', choices=['zoom', 'brightness'], type=str)
parser.add_argument('--trans_arg', default=0.5, type=float)
parser.add_argument('--train_batch_size', default=100, type=int)
parser.add_argument('--test_batch_size', default=100, type=int)
parser.add_argument('--seed', default=100, type=int)
parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda'])


def write_summaries(summaries, dataset_name, model_name, cal_method, trans_type, trans_arg, mode):
  summaries_dir = '../../summaries_onetime'
  if not os.path.exists(summaries_dir):
    os.makedirs(summaries_dir)

  fn = os.path.join(summaries_dir, 'summaries_{}_{}_{}_{}_{}_{}.csv'.format(dataset_name,
                                                                            model_name,
                                                                            cal_method,
                                                                            trans_type,
                                                                            trans_arg,
                                                                            mode))
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


def main(args):
  if args.seed is not None:
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)

  device = torch.device('cuda' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu')

  with open('logging.json', 'rt') as f:
    config = json.load(f)

  logging.config.dictConfig(config)
  logger = logging.getLogger()
  logger.info('Started to load dataset')
  stored_val_idx = None
  if args.dataset_name == 'MNIST':
    dataset = MNIST()
    classifier = MNIST_LeNet5(device=device)
    classifier.init(False)
  elif args.dataset_name == 'CIFAR10':
    dataset = CIFAR(is_ten=True)
    classifier = CIFARModel(device=device, model_name=args.model_name, is_ten=True)
    stored_val_idx = classifier.init(False)
  elif args.dataset_name == 'CIFAR100':
    dataset = CIFAR(is_ten=False)
    classifier = CIFARModel(device=device, model_name=args.model_name, is_ten=False)
    stored_val_idx = classifier.init(False)
  elif args.dataset_name == 'ImageNet':
    dataset = ImageNet('~/datasets/imagenet/')
    classifier = ImageNet_Model(device=device, model_name=args.model_name)
    classifier.init(False)

  output_dir = os.path.join('outputs_onetime',
                            args.dataset_name,
                            '{}_{}'.format(args.trans_type, args.trans_arg),
                            args.model_name,
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

  if args.after_ts:
    print('start to calibrate')
    ts_start_time = time.time()
    classifier.run_temp_scaling(val_loader)
    print('took {:.2f} secs'.format(time.time() - ts_start_time))
    print('Temperature', classifier.ts_network.temp)

  trans = [('identity', 1.0), (args.trans_type, args.trans_arg)]
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
    test_output_all, test_pred_all, test_true_all, test_conf_all = \
      classifier.compute_output(classifier.ts_network if args.after_ts else classifier.network, trans_test_loader)
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
  ###

  post_cal = PostCalTransPool(args.dataset_name,
                              args.model_name,
                              'ts' if args.after_ts else 'uncalibrated',
                              output_dir,
                              1,
                              'one_time',
                              device)

  logger.info('started to calibrate')
  start = time.time()
  val_summaries, stop_idx, before_accuracy, after_accuracy = post_cal.calibrate()
  write_summaries(val_summaries, args.dataset_name, args.model_name,
                  'ts' if args.after_ts else 'uncalibrated', args.trans_type, args.trans_arg, 'val')
  logger.info('finished. took {:.2f} secs'.format(time.time() - start))
  logger.info('Val Accuracy. Before : {:.5f}, After: {:.5f}'.format(before_accuracy, after_accuracy))

  logger.info('started to test')
  start = time.time()
  test_summaries, final_ece, before_accuracy, after_accuracy = post_cal.test(stop_idx)
  write_summaries(test_summaries, args.dataset_name, args.model_name,
                  'ts' if args.after_ts else 'uncalibrated', args.trans_type, args.trans_arg, 'test')

  logger.info('finished. took {:.2f} secs'.format(time.time() - start))
  logger.info('Test Accuracy. Before : {:.5f}, After: {:.5f}'.format(before_accuracy, after_accuracy))
  logger.info('Final ECE: {:.5f}, MCE: {:.5f}, VCE: {:.5f} at iter {}'.format(final_ece.compute_ECE(),
                                                                              final_ece.compute_MCE(),
                                                                              final_ece.compute_VCE(),
                                                                              stop_idx))


if __name__ == '__main__':
  args = parser.parse_args()

  main(args)
