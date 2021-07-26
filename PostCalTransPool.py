import logging
import glob

import os
import time

import numpy as np
from sklearn.metrics import log_loss

import torch
import torch.nn.functional as F

from random import choices

from Ece import ECE


class PostCalTransPool:
  def __init__(self, dataset, model_name, cal_method, root_dir, L, iter_list_str, device):
    self.dataset = dataset
    self.model_name = model_name
    self.cal_method = cal_method
    self.root_dir = root_dir
    self.L = L
    self.iter_list_str = iter_list_str
    self.device = device

    self.trans_list = []
    self.pairs = []
    self.ori_outputs = {'val': {}, 'test': {}}
    self.ori_trues = {'val': {}, 'test': {}}
    self.transformed_result = {'val': {}, 'test': {}}

    self.temps_ori = []
    self.temps2 = []

    self.logger = logging.getLogger(self.__class__.__qualname__)

  def split_name(self, name):
    arr = name.split('_')
    tran_type, tran_arg, val_test, data_kind = arr[3], arr[5], arr[6], arr[7]
    return tran_type, tran_arg, val_test, data_kind

  def _load_files(self, root_dir):
    for name in glob.glob(root_dir + "/*.npy"):
      tran_type, tran_arg, val_test, data_kind = self.split_name(os.path.splitext(os.path.basename(name))[0])
      data = np.load(name)

      data = torch.from_numpy(data)

      if tran_type == 'identity':
        if data_kind == 'y':
          self.ori_trues[val_test] = data
        elif data_kind == 'logits':
          self.ori_outputs[val_test] = data
      else:
        if data_kind == 'logits':
          self.transformed_result[val_test][(tran_type, tran_arg,)] = data

      if val_test == 'val' and not tran_type == 'identity' and data_kind == 'y':
        self.trans_list.append((tran_type, tran_arg))

    self.trans_list = sorted(self.trans_list)

    print('Number of transformation: {}'.format(len(self.trans_list)))

  def _sample_pairs(self, include_original=True):
    if include_original:
      self.pairs = choices(self.trans_list, k=self.L)
      self.pairs = list(zip([('identity', 1)] * self.L, self.pairs))
    else:
      pass

    return self.pairs

  def _calibrate(self, conf, true_idx, tol=1e-6, max_iter=30, num_guess=100):
    conf = conf.transpose()
    conf = np.maximum(conf, 1e-16)
    x = np.log(conf)

    # xt = np.zeros((1, x.shape[1]))
    # for j in range(x.shape[1]):
    #   xt[0, j] = x[pred[j], j]

    xt = x[true_idx, np.arange(x.shape[1])]
    xt = np.expand_dims(xt, axis=0)

    cal = np.linspace(start=0.1, stop=10, num=num_guess)

    for j in range(len(cal)):
      for n in range(max_iter):
        f1 = np.sum(xt - np.divide(np.sum(np.multiply(x, np.exp(cal[j] * x)), 0), np.sum(np.exp(cal[j] * x), 0)))
        f2 = np.sum(np.divide(-np.sum(np.multiply(np.square(x), np.exp(cal[j] * x)), 0),
                              np.sum(np.exp(cal[j] * x), 0))
                    + np.divide(np.square(np.sum(np.multiply(x, np.exp(cal[j] * x)), 0)),
                                np.square(np.sum(np.exp(cal[j] * x), 0))))

        cal[j] = cal[j] - f1 / f2
        if np.isnan(f1) or np.isnan(f2):
          break
        if np.abs(f1 / f2) < tol:
          break
    cal = np.append([1], cal, axis=0)
    f0 = np.zeros(cal.shape)
    for j in range(len(cal)):
      f0[j] = np.sum(-np.log(np.divide(np.exp(cal[j] * xt), np.sum(np.exp(cal[j] * x), 0))), 1)

    n = np.nanargmin(f0)
    cal = cal[n]
    if n == 0:
      print("calibration failed")

    return cal

  def _calibrate_part(self, conf_all, true_all):
    temp = self._calibrate(conf_all, true_all)
    self.logger.info('Temperature: {:.4f}'.format(temp))
    return temp

  def _temp_scale(self, logits, t):
    temp = t.unsqueeze(1).expand(logits.size(0), logits.size(1))
    conf = logits
    # conf = F.softmax(logits, dim=1)
    conf = torch.clamp(conf, min=1e-16)
    return torch.log(conf) * temp

  def compute_case_ece(self, idx, pred_all, true_all, conf_pred_all):
    pred_arr = pred_all[idx]
    true_arr = true_all[idx]
    conf_pred_arr = conf_pred_all[idx]

    ece = ECE(15)
    ece.add_data(pred_arr, true_arr, conf_pred_arr)
    return ece

  def _get_data(self, mode, tran):
    if tran[0] == 'identity':
      data = self.ori_outputs[mode]
    else:
      data = self.transformed_result[mode][tran]
    return data

  def _get_pred_conf(self, mode, tran):
    data = self._get_data(mode, tran)

    pred = data.data.max(1, keepdim=True)[1].squeeze()
    conf = F.softmax(data, dim=1)
    conf_pred = conf.gather(1, pred.view(-1, 1)).squeeze()

    return data, pred, conf, conf_pred

  def _categorize(self, pred_1, conf_pred_1, pred_2, conf_pred_2):
    pred_eq_idx = pred_1 == pred_2
    pred_neq_idx = pred_1 != pred_2

    conf_incr_idx = conf_pred_1 < conf_pred_2
    conf_not_incr_idx = conf_pred_1 >= conf_pred_2

    case_1_idx = pred_neq_idx & conf_incr_idx
    case_2_idx = pred_neq_idx & conf_not_incr_idx
    case_3_idx = pred_eq_idx & conf_incr_idx
    case_4_idx = pred_eq_idx & conf_not_incr_idx

    return case_1_idx, case_2_idx, case_3_idx, case_4_idx

  def _modified_temp(self, n_total, n_case_data, raw_sigma):
    ratio = torch.true_divide(n_case_data, n_total)
    return (1 - ratio) * 1 + ratio * raw_sigma

  def _cal(self, confs, trues, case_1_idx, case_2_idx, case_3_idx, case_4_idx):
    res = {'case_1': {'temp': torch.ones(1, requires_grad=False)},
           'case_2': {'temp': torch.ones(1, requires_grad=False)},
           'case_3': {'temp': torch.ones(1, requires_grad=False)},
           'case_4': {'temp': torch.ones(1, requires_grad=False)}}

    n_case_1 = case_1_idx.sum()
    n_case_2 = case_2_idx.sum()
    n_case_3 = case_3_idx.sum()
    n_case_4 = case_4_idx.sum()
    n_total = n_case_1 + n_case_2 + n_case_3 + n_case_4
    if n_case_1 > 0:
      raw_temp = self._calibrate_part(confs[case_1_idx].numpy(), trues[case_1_idx].numpy())
      res['case_1']['temp'] *= self._modified_temp(n_total, n_case_1, raw_temp)
    if case_2_idx.sum() > 0:
      raw_temp = self._calibrate_part(confs[case_2_idx].numpy(), trues[case_2_idx].numpy())
      res['case_2']['temp'] *= self._modified_temp(n_total, n_case_2, raw_temp)
    if case_3_idx.sum() > 0:
      raw_temp = self._calibrate_part(confs[case_3_idx].numpy(), trues[case_3_idx].numpy())
      res['case_3']['temp'] *= self._modified_temp(n_total, n_case_3, raw_temp)
    if case_4_idx.sum() > 0:
      raw_temp = self._calibrate_part(confs[case_4_idx].numpy(), trues[case_4_idx].numpy())
      res['case_4']['temp'] *= self._modified_temp(n_total, n_case_4, raw_temp)

    return res

  def _update_tran_res(self, mode, tran, res):
    dest = self.ori_outputs[mode] if tran[0] == 'identity' else self.transformed_result[mode][tran]

    case_1_idx = res['case_1']['idx']
    case_2_idx = res['case_2']['idx']
    case_3_idx = res['case_3']['idx']
    case_4_idx = res['case_4']['idx']

    if case_1_idx.sum() > 0:
      dest[case_1_idx] = res['case_1']['scaled']
    if case_2_idx.sum() > 0:
      dest[case_2_idx] = res['case_2']['scaled']
    if case_3_idx.sum() > 0:
      dest[case_3_idx] = res['case_3']['scaled']
    if case_4_idx.sum() > 0:
      dest[case_4_idx] = res['case_4']['scaled']

  def _apply_ts_and_update(self, temps, conf, dest, indices):
    for case_num in indices:
      case_idx = indices[case_num]

      scaled = self._temp_scale(conf[case_idx], temps[case_num-1])

      dest[case_idx] = scaled

  def calibrate(self, stop_thr=1e-5):
    self.logger.info('started to calibrate')
    start = time.time()

    mode = 'val'
    self._load_files(self.root_dir)

    assert len(self.trans_list) > 0

    self._sample_pairs(self.L)
    summaries = []
    before_accuracy = (self.ori_outputs['val'].argmax(dim=1) == self.ori_trues['val']).float().mean().item()
    pre_ece = 1.0
    stop_idx = None
    for i, (trans_1, trans_2) in enumerate(self.pairs):
      print(i+1, trans_1, trans_2)
      summary = {'iter': i+1,
                 'trans_1_type': trans_1[0], 'trans_1_arg': trans_1[1],
                 'trans_2_type': trans_2[0], 'trans_2_arg': trans_2[1]}

      data_1, pred_1, conf_1, conf_pred_1 = self._get_pred_conf(mode, trans_1)
      data_2, pred_2, conf_2, conf_pred_2 = self._get_pred_conf(mode, trans_2)

      # use the confidence of original predicted label
      conf_pred_2_old_pred = conf_2.gather(1, pred_1.view(-1, 1)).squeeze()

      case_1_idx, case_2_idx, case_3_idx, case_4_idx = self._categorize(pred_1, conf_pred_1, pred_2, conf_pred_2_old_pred)

      before_ece = ECE(15)
      before_ece.add_data(pred_1.numpy(), self.ori_trues[mode].numpy(), conf_pred_1.numpy())

      self.logger.info('Before ECE: {:.6f}'.format(before_ece.compute_ECE()))
      self.logger.info('Before MCE: {:.6f}'.format(before_ece.compute_MCE()))
      self.logger.info('Before VCE: {:.6f}'.format(before_ece.compute_VCE()))

      summary['trans_1_before_ece'] = before_ece.compute_ECE()
      summary['trans_1_before_mce'] = before_ece.compute_MCE()
      summary['trans_1_before_vce'] = before_ece.compute_VCE()

      summary['trans_1_before_nll'] = log_loss(y_true=self.ori_trues[mode].numpy(), y_pred=conf_1.numpy())

      before_case1_ece = self.compute_case_ece(case_1_idx, pred_1.numpy(), self.ori_trues[mode].numpy(), conf_pred_1.numpy())
      before_case2_ece = self.compute_case_ece(case_2_idx, pred_1.numpy(), self.ori_trues[mode].numpy(), conf_pred_1.numpy())
      before_case3_ece = self.compute_case_ece(case_3_idx, pred_1.numpy(), self.ori_trues[mode].numpy(), conf_pred_1.numpy())
      before_case4_ece = self.compute_case_ece(case_4_idx, pred_1.numpy(), self.ori_trues[mode].numpy(), conf_pred_1.numpy())

      self.logger.info(
        'Before ECE by case: {:.6f}, {:.6f}, {:.6f}, {:.6f}'.format(before_case1_ece.compute_ECE(),
                                                                    before_case2_ece.compute_ECE(),
                                                                    before_case3_ece.compute_ECE(),
                                                                    before_case4_ece.compute_ECE()))
      self.logger.info(
        'Before MCE by case: {:.6f}, {:.6f}, {:.6f}, {:.6f}'.format(before_case1_ece.compute_MCE(),
                                                                    before_case2_ece.compute_MCE(),
                                                                    before_case3_ece.compute_MCE(),
                                                                    before_case4_ece.compute_MCE()))
      self.logger.info(
        'Before VCE by case: {:.6f}, {:.6f}, {:.6f}, {:.6f}'.format(before_case1_ece.compute_VCE(),
                                                                    before_case2_ece.compute_VCE(),
                                                                    before_case3_ece.compute_VCE(),
                                                                    before_case4_ece.compute_VCE()))

      cal_res = self._cal(conf_1, self.ori_trues[mode], case_1_idx, case_2_idx, case_3_idx, case_4_idx)
      cal_res['case_1']['idx'] = case_1_idx
      cal_res['case_2']['idx'] = case_2_idx
      cal_res['case_3']['idx'] = case_3_idx
      cal_res['case_4']['idx'] = case_4_idx

      self.logger.info('Temperature: {:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(cal_res['case_1']['temp'].item(),
                                                                            cal_res['case_2']['temp'].item(),
                                                                            cal_res['case_3']['temp'].item(),
                                                                            cal_res['case_4']['temp'].item()))

      #############

      def _compute_ece(logits, true_all):
        ece = ECE(15)
        pred = logits.argmax(dim=1)
        conf = F.softmax(logits, dim=1)
        conf_pred = conf.gather(1, pred.view(-1, 1)).squeeze()

        ece.add_data(pred.numpy(), true_all.numpy(), conf_pred.numpy())
        return ece

      increase = False
      calibrated = conf_1.clone()
      calibrated[case_1_idx] = self._temp_scale(conf_1[case_1_idx], cal_res['case_1']['temp'])
      calibrated[case_2_idx] = self._temp_scale(conf_1[case_2_idx], cal_res['case_2']['temp'])
      calibrated[case_3_idx] = self._temp_scale(conf_1[case_3_idx], cal_res['case_3']['temp'])
      calibrated[case_4_idx] = self._temp_scale(conf_1[case_4_idx], cal_res['case_4']['temp'])
      after_ece = _compute_ece(calibrated, self.ori_trues[mode])
      if after_ece.compute_ECE() > before_ece.compute_ECE():
        print('ECE increased from {:.6f} to {:.6f}. set all temps to 1.'.format(before_ece.compute_ECE(), after_ece.compute_ECE()))
        cal_res['case_1']['temp'] = torch.ones(1, requires_grad=False)
        cal_res['case_2']['temp'] = torch.ones(1, requires_grad=False)
        cal_res['case_3']['temp'] = torch.ones(1, requires_grad=False)
        cal_res['case_4']['temp'] = torch.ones(1, requires_grad=False)
        increase = True

      def _guard_temp(temp, conf_case, trues_case, before_case_ece):
        if abs(temp.item() - 1.0000) <= 1e-10:
          return temp

        scaled = self._temp_scale(conf_case, temp)
        after_case_ece = _compute_ece(scaled, trues_case)
        if before_case_ece.compute_ECE() >= after_case_ece.compute_ECE():
          return temp
        else:
          self.logger.info('Case ECE increased. Change temperature from {:.6f} to 1.'.format(temp.item()))
          return torch.ones(1, requires_grad=False)

      cal_res['case_1']['temp'] = _guard_temp(cal_res['case_1']['temp'], conf_1[case_1_idx], self.ori_trues[mode][case_1_idx], before_case1_ece)
      cal_res['case_2']['temp'] = _guard_temp(cal_res['case_2']['temp'], conf_1[case_2_idx], self.ori_trues[mode][case_2_idx], before_case2_ece)
      cal_res['case_3']['temp'] = _guard_temp(cal_res['case_3']['temp'], conf_1[case_3_idx], self.ori_trues[mode][case_3_idx], before_case3_ece)
      cal_res['case_4']['temp'] = _guard_temp(cal_res['case_4']['temp'], conf_1[case_4_idx], self.ori_trues[mode][case_4_idx], before_case4_ece)


      #############
      self.temps_ori.append((cal_res['case_1']['temp'],
                             cal_res['case_2']['temp'],
                             cal_res['case_3']['temp'],
                             cal_res['case_4']['temp'],))

      summary['trans_1_case_1_temp'] = cal_res['case_1']['temp'].item()
      summary['trans_1_case_2_temp'] = cal_res['case_2']['temp'].item()
      summary['trans_1_case_3_temp'] = cal_res['case_3']['temp'].item()
      summary['trans_1_case_4_temp'] = cal_res['case_4']['temp'].item()

      dst_1 = self._get_data(mode, trans_1)
      case_indices = {}
      if case_1_idx.sum() > 0:
        case_indices[1] = case_1_idx
      if case_2_idx.sum() > 0:
        case_indices[2] = case_2_idx
      if case_3_idx.sum() > 0:
        case_indices[3] = case_3_idx
      if case_4_idx.sum() > 0:
        case_indices[4] = case_4_idx

      self._apply_ts_and_update(self.temps_ori[i], conf_1, dst_1, case_indices)

      before_ece = ECE(15)
      before_ece.add_data(pred_2.numpy(), self.ori_trues[mode].numpy(), conf_pred_2.numpy())
      summary['trans_2_before_ece'] = before_ece.compute_ECE()
      summary['trans_2_before_mce'] = before_ece.compute_MCE()
      summary['trans_2_before_vce'] = before_ece.compute_VCE()

      after_ece = ECE(15)
      pred_temp = data_1.data.max(1, keepdim=True)[1].squeeze()
      conf_all_temp = F.softmax(data_1, dim=1)
      conf_pred_temp = conf_all_temp.gather(1, pred_temp.view(-1, 1)).squeeze()
      after_ece.add_data(pred_temp.numpy(), self.ori_trues[mode].numpy(), conf_pred_temp.numpy())

      self.logger.info('After ECE: {:.6f}'.format(after_ece.compute_ECE()))
      self.logger.info('After MCE: {:.6f}'.format(after_ece.compute_MCE()))
      self.logger.info('After VCE: {:.6f}'.format(after_ece.compute_VCE()))

      print('\t', case_1_idx.sum().item(), case_2_idx.sum().item(), case_3_idx.sum().item(), case_4_idx.sum().item())

      if stop_idx is None and not increase and np.abs(after_ece.compute_ECE() - pre_ece) < stop_thr:
        time_passed = time.time() - start
        stop_idx = i+1
        after_accuracy = (self.ori_outputs['val'].argmax(dim=1) == self.ori_trues['val']).float().mean().item()
        self.logger.info("ECE change is less than threshold. {:.6f} < {:.6f}".format(np.abs(after_ece.compute_ECE() - pre_ece),
                                                                                     stop_thr))
        self.logger.info("Time elapsed: {:.2f} secs".format(time_passed))

      summary['trans_1_after_ece'] = after_ece.compute_ECE()
      summary['trans_1_after_mce'] = after_ece.compute_MCE()
      summary['trans_1_after_vce'] = after_ece.compute_VCE()

      summary['trans_1_after_nll'] = log_loss(y_true=self.ori_trues[mode].numpy(), y_pred=conf_all_temp.numpy())

      summary['trans_1_case_1_cnt'] = case_1_idx.sum().item()
      summary['trans_1_case_2_cnt'] = case_2_idx.sum().item()
      summary['trans_1_case_3_cnt'] = case_3_idx.sum().item()
      summary['trans_1_case_4_cnt'] = case_4_idx.sum().item()

      after_case1_ece = self.compute_case_ece(case_1_idx, pred_1.numpy(), self.ori_trues[mode].numpy(), conf_pred_temp.numpy())
      after_case2_ece = self.compute_case_ece(case_2_idx, pred_1.numpy(), self.ori_trues[mode].numpy(), conf_pred_temp.numpy())
      after_case3_ece = self.compute_case_ece(case_3_idx, pred_1.numpy(), self.ori_trues[mode].numpy(), conf_pred_temp.numpy())
      after_case4_ece = self.compute_case_ece(case_4_idx, pred_1.numpy(), self.ori_trues[mode].numpy(), conf_pred_temp.numpy())
      self.logger.info(
        'after ECE by case: {:.6f}, {:.6f}, {:.6f}, {:.6f}'.format(after_case1_ece.compute_ECE(),
                                                                   after_case2_ece.compute_ECE(),
                                                                   after_case3_ece.compute_ECE(),
                                                                   after_case4_ece.compute_ECE(), ))
      self.logger.info(
        'after MCE by case: {:.6f}, {:.6f}, {:.6f}, {:.6f}'.format(after_case1_ece.compute_MCE(),
                                                                   after_case2_ece.compute_MCE(),
                                                                   after_case3_ece.compute_MCE(),
                                                                   after_case4_ece.compute_MCE(), ))
      self.logger.info(
        'after VCE by case: {:.6f}, {:.6f}, {:.6f}, {:.6f}'.format(after_case1_ece.compute_VCE(),
                                                                   after_case2_ece.compute_VCE(),
                                                                   after_case3_ece.compute_VCE(),
                                                                   after_case4_ece.compute_VCE(), ))

      pre_ece = after_ece.compute_ECE()

      # Second elem
      self.logger.info('Second Element')

      dst_2 = self._get_data(mode, trans_2)
      self._apply_ts_and_update(self.temps_ori[i], conf_2, dst_2, case_indices)

      after_ece = ECE(15)
      pred_temp = data_2.data.max(1, keepdim=True)[1].squeeze()
      conf_all_temp = F.softmax(data_2, dim=1)
      conf_pred_temp = conf_all_temp.gather(1, pred_temp.view(-1, 1)).squeeze()
      after_ece.add_data(pred_temp.numpy(), self.ori_trues[mode].numpy(), conf_pred_temp.numpy())

      self.logger.info('After ECE: {:.6f}'.format(after_ece.compute_ECE()))
      self.logger.info('After MCE: {:.6f}'.format(after_ece.compute_MCE()))
      self.logger.info('After VCE: {:.6f}'.format(after_ece.compute_VCE()))
      print('\t', case_1_idx.sum().item(), case_2_idx.sum().item(), case_3_idx.sum().item(), case_4_idx.sum().item())

      summary['trans_2_after_ece'] = after_ece.compute_ECE()
      summary['trans_2_after_mce'] = after_ece.compute_MCE()
      summary['trans_2_after_vce'] = after_ece.compute_VCE()

      summary['trans_2_case_1_cnt'] = case_1_idx.sum().item()
      summary['trans_2_case_2_cnt'] = case_2_idx.sum().item()
      summary['trans_2_case_3_cnt'] = case_3_idx.sum().item()
      summary['trans_2_case_4_cnt'] = case_4_idx.sum().item()

      summaries.append(summary)

    time_pass = time.time() - start
    self.logger.info('Finished calibrating. It took {:.2f} secs'.format(time_pass))

    if stop_idx is None:
      self.logger.info('No Stop Index is set. Set to {}'.format(len(self.pairs)))
      stop_idx = len(self.pairs)
      after_accuracy = (self.ori_outputs['val'].argmax(dim=1) == self.ori_trues['val']).float().mean().item()

    return summaries, stop_idx, before_accuracy, after_accuracy

  def test(self, stop_idx):
    self.logger.info('started to test')
    start = time.time()

    mode = 'test'
    summaries = []
    before_accuracy = (self.ori_outputs['test'].argmax(dim=1) == self.ori_trues['test']).float().mean().item()
    for i, (trans_1, trans_2) in enumerate(self.pairs):
      print(i+1, trans_1, trans_2)

      summary = {'iter': i + 1,
                 'trans_1_type': trans_1[0], 'trans_1_arg': trans_1[1],
                 'trans_2_type': trans_2[0], 'trans_2_arg': trans_2[1]}

      data_1, pred_1, conf_1, conf_pred_1 = self._get_pred_conf(mode, trans_1)
      data_2, pred_2, conf_2, conf_pred_2 = self._get_pred_conf(mode, trans_2)

      # use the confidence of original predicted label
      conf_pred_2_old_pred = conf_2.gather(1, pred_1.view(-1, 1)).squeeze()

      case_1_idx, case_2_idx, case_3_idx, case_4_idx = self._categorize(pred_1, conf_pred_1, pred_2, conf_pred_2_old_pred)

      before_ece = ECE(15)
      before_ece.add_data(pred_1.numpy(), self.ori_trues[mode].numpy(), conf_pred_1.numpy())
      self.logger.info('Before ECE: {:.6f}'.format(before_ece.compute_ECE()))
      self.logger.info('Before MCE: {:.6f}'.format(before_ece.compute_MCE()))
      self.logger.info('Before VCE: {:.6f}'.format(before_ece.compute_VCE()))

      summary['trans_1_before_ece'] = before_ece.compute_ECE()
      summary['trans_1_before_mce'] = before_ece.compute_MCE()
      summary['trans_1_before_vce'] = before_ece.compute_VCE()

      summary['trans_1_before_nll'] = log_loss(y_true=self.ori_trues[mode].numpy(), y_pred=conf_1.numpy())

      before_case1_ece = self.compute_case_ece(case_1_idx, pred_1.numpy(), self.ori_trues[mode].numpy(),
                                               conf_pred_1.numpy())
      before_case2_ece = self.compute_case_ece(case_2_idx, pred_1.numpy(), self.ori_trues[mode].numpy(),
                                               conf_pred_1.numpy())
      before_case3_ece = self.compute_case_ece(case_3_idx, pred_1.numpy(), self.ori_trues[mode].numpy(),
                                               conf_pred_1.numpy())
      before_case4_ece = self.compute_case_ece(case_4_idx, pred_1.numpy(), self.ori_trues[mode].numpy(),
                                               conf_pred_1.numpy())
      self.logger.info(
        'Before ECE by case: {:.6f}, {:.6f}, {:.6f}, {:.6f}'.format(before_case1_ece.compute_ECE(),
                                                                    before_case2_ece.compute_ECE(),
                                                                    before_case3_ece.compute_ECE(),
                                                                    before_case4_ece.compute_ECE(), ))
      self.logger.info(
        'Before MCE by case: {:.6f}, {:.6f}, {:.6f}, {:.6f}'.format(before_case1_ece.compute_MCE(),
                                                                    before_case2_ece.compute_MCE(),
                                                                    before_case3_ece.compute_MCE(),
                                                                    before_case4_ece.compute_MCE(), ))
      self.logger.info(
        'Before VCE by case: {:.6f}, {:.6f}, {:.6f}, {:.6f}'.format(before_case1_ece.compute_VCE(),
                                                                    before_case2_ece.compute_VCE(),
                                                                    before_case3_ece.compute_VCE(),
                                                                    before_case4_ece.compute_VCE(), ))

      summary['trans_1_case_1_temp'] = self.temps_ori[i][0].item()
      summary['trans_1_case_2_temp'] = self.temps_ori[i][1].item()
      summary['trans_1_case_3_temp'] = self.temps_ori[i][2].item()
      summary['trans_1_case_4_temp'] = self.temps_ori[i][3].item()

      before_ece = ECE(15)
      before_ece.add_data(pred_2.numpy(), self.ori_trues[mode].numpy(), conf_pred_2.numpy())
      summary['trans_2_before_ece'] = before_ece.compute_ECE()
      summary['trans_2_before_mce'] = before_ece.compute_MCE()
      summary['trans_2_before_vce'] = before_ece.compute_VCE()

      dst_1 = self._get_data(mode, trans_1)
      case_indices = {}
      if case_1_idx.sum() > 0:
        case_indices[1] = case_1_idx
      if case_2_idx.sum() > 0:
        case_indices[2] = case_2_idx
      if case_3_idx.sum() > 0:
        case_indices[3] = case_3_idx
      if case_4_idx.sum() > 0:
        case_indices[4] = case_4_idx
      self._apply_ts_and_update(self.temps_ori[i], conf_1, dst_1, case_indices)

      after_ece = ECE(15)
      pred_temp = data_1.data.max(1, keepdim=True)[1].squeeze()
      conf_all_temp = F.softmax(data_1, dim=1)
      conf_pred_temp = conf_all_temp.gather(1, pred_temp.view(-1, 1)).squeeze()

      after_ece.add_data(pred_temp.numpy(), self.ori_trues[mode].numpy(), conf_pred_temp.numpy())

      case_1_ece = self.compute_case_ece(case_1_idx, pred_1.numpy(), self.ori_trues[mode].numpy(), conf_pred_temp.numpy())
      case_2_ece = self.compute_case_ece(case_2_idx, pred_1.numpy(), self.ori_trues[mode].numpy(), conf_pred_temp.numpy())
      case_3_ece = self.compute_case_ece(case_3_idx, pred_1.numpy(), self.ori_trues[mode].numpy(), conf_pred_temp.numpy())
      case_4_ece = self.compute_case_ece(case_4_idx, pred_1.numpy(), self.ori_trues[mode].numpy(), conf_pred_temp.numpy())
      print('after ece by case {:.6f}, {:.6f}, {:.6f}, {:.6f}'.format(case_1_ece.compute_ECE(),
                                                                      case_2_ece.compute_ECE(),
                                                                      case_3_ece.compute_ECE(),
                                                                      case_4_ece.compute_ECE()))
      print('after mce by case {:.6f}, {:.6f}, {:.6f}, {:.6f}'.format(case_1_ece.compute_MCE(),
                                                                      case_2_ece.compute_MCE(),
                                                                      case_3_ece.compute_MCE(),
                                                                      case_4_ece.compute_MCE()))
      print('after vce by case {:.6f}, {:.6f}, {:.6f}, {:.6f}'.format(case_1_ece.compute_VCE(),
                                                                      case_2_ece.compute_VCE(),
                                                                      case_3_ece.compute_VCE(),
                                                                      case_4_ece.compute_VCE()))

      self.logger.info('After ECE: {:.6f}'.format(after_ece.compute_ECE()))
      self.logger.info('After MCE: {:.6f}'.format(after_ece.compute_MCE()))
      self.logger.info('After VCE: {:.6f}'.format(after_ece.compute_VCE()))

      print('\t', case_1_idx.sum().item(), case_2_idx.sum().item(), case_3_idx.sum().item(), case_4_idx.sum().item())

      if i+1 == stop_idx:
        time_passed = time.time() - start
        final_ece = after_ece

        if self.iter_list_str == 'one_time':
          output_dir = './outputs_onetime/{}/ours/{}/{}'.format(self.dataset,
                                                                '_'.join(self.trans_list[0]),
                                                                self.cal_method)
        else:
          output_dir = './outputs/{}/ours/{}_N_{}/{}'.format(self.dataset,
                                                             self.iter_list_str,
                                                             len(self.trans_list),
                                                             self.cal_method)
        if not os.path.exists(output_dir):
          os.makedirs(output_dir)

        print('stored to {}'.format(output_dir))

        fn_logits = '{}_{}_type_identity_arg_1.0_val_logits.npy'.format(self.dataset, self.model_name,)
        fn_y = '{}_{}_type_identity_arg_1.0_val_y.npy'.format(self.dataset, self.model_name)
        np.save(os.path.join(output_dir, fn_logits), self.ori_outputs['val'].numpy())
        np.save(os.path.join(output_dir, fn_y), self.ori_trues['val'].numpy())

        fn_logits = '{}_{}_type_identity_arg_1.0_test_logits.npy'.format(self.dataset, self.model_name)
        fn_y = '{}_{}_type_identity_arg_1.0_test_y.npy'.format(self.dataset, self.model_name)
        np.save(os.path.join(output_dir, fn_logits), self.ori_outputs['test'].numpy())
        np.save(os.path.join(output_dir, fn_y), self.ori_trues['test'].numpy())

        after_accuracy = (self.ori_outputs['test'].argmax(dim=1) == self.ori_trues['test']).float().mean().item()
        self.logger.info("Time Elapsed: {:.2f} secs".format(time_passed))

      summary['trans_1_after_ece'] = after_ece.compute_ECE()
      summary['trans_1_after_mce'] = after_ece.compute_MCE()
      summary['trans_1_after_vce'] = after_ece.compute_VCE()

      summary['trans_1_case_1_cnt'] = case_1_idx.sum().item()
      summary['trans_1_case_2_cnt'] = case_2_idx.sum().item()
      summary['trans_1_case_3_cnt'] = case_3_idx.sum().item()
      summary['trans_1_case_4_cnt'] = case_4_idx.sum().item()

      summary['trans_1_after_nll'] = log_loss(y_true=self.ori_trues[mode].numpy(), y_pred=conf_all_temp.numpy())

      dst_2 = self._get_data(mode, trans_2)
      self._apply_ts_and_update(self.temps_ori[i], conf_2, dst_2, case_indices)

      after_ece = ECE(15)
      pred_temp = data_2.data.max(1, keepdim=True)[1].squeeze()
      conf_all_temp = F.softmax(data_2, dim=1)
      conf_pred_temp = conf_all_temp.gather(1, pred_temp.view(-1, 1)).squeeze()

      after_ece.add_data(pred_temp.numpy(), self.ori_trues[mode].numpy(), conf_pred_temp.numpy())

      summary['trans_2_after_ece'] = after_ece.compute_ECE()
      summary['trans_2_after_mce'] = after_ece.compute_MCE()
      summary['trans_2_after_vce'] = after_ece.compute_VCE()

      summary['trans_2_case_1_cnt'] = case_1_idx.sum().item()
      summary['trans_2_case_2_cnt'] = case_2_idx.sum().item()
      summary['trans_2_case_3_cnt'] = case_3_idx.sum().item()
      summary['trans_2_case_4_cnt'] = case_4_idx.sum().item()

      summaries.append(summary)

    time_pass = time.time() - start
    self.logger.info('Finished calibrating. It took {} secs'.format(time_pass))
    return summaries, final_ece, before_accuracy, after_accuracy
