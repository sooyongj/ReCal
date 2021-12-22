import glob
import os
import numpy as np
import matplotlib.pyplot as plt

from scipy.special import softmax

from ReCal.Ece import ECE


def _extract_conf_pred(logits):
  pred = np.argmax(logits, axis=1)
  conf = softmax(logits, axis=1)
  conf_pred = conf[np.arange(conf.shape[0]), pred]
  return pred, conf, conf_pred


def _compute_ece(pred, y, conf):
  ece = ECE(15)
  ece.add_data(pred, y, conf)
  return ece.compute_ECE()


def _compute_brier(conf, y):
  targets = np.zeros_like(conf)
  targets[np.arange(y.size), y] = 1

  return np.average(np.sum((conf - targets) ** 2, axis=1)) / conf.shape[1]  #??


def analyze_dir(dataset, model, cal_method, dir):
  folder = dir.split('/')[-1]
  tran_type = folder.split('_')[0]
  tran_arg = folder.split('_')[1]

  logits = np.load(os.path.join(dir, cal_method, '{}_{}_type_identity_arg_1.0_test_logits.npy'.format(dataset, model)))
  y = np.load(os.path.join(dir, cal_method, '{}_{}_type_identity_arg_1.0_test_y.npy'.format(dataset, model)))

  tran_logits = np.load(os.path.join(dir, cal_method, '{}_{}_type_{}_arg_{}_test_logits.npy'.format(dataset,
                                                                                                    model,
                                                                                                    tran_type,
                                                                                                    tran_arg)))
  pred, conf, conf_pred = _extract_conf_pred(logits)
  tran_pred, tran_conf, tran_conf_pred = _extract_conf_pred(tran_logits)

  pred_chg = pred != tran_pred
  pred_no_chg = pred == tran_pred
  conf_not_inc = conf_pred >= tran_conf_pred
  conf_inc = conf_pred < tran_conf_pred

  case_1_idx = pred_chg & conf_inc
  case_2_idx = pred_chg & conf_not_inc
  case_3_idx = pred_no_chg & conf_inc
  case_4_idx = pred_no_chg & conf_not_inc

  res = {'acc': (pred == y).mean(), 'ece': _compute_ece(pred, y, conf_pred), 'cnt': len(y), 'brier': _compute_brier(conf, y),
         'case_1': {}, 'case_2': {}, 'case_3': {}, 'case_4': {}}

  res['case_1']['count'] = np.sum(case_1_idx)
  res['case_1']['accuracy'] = (pred[case_1_idx] == y[case_1_idx]).mean()
  res['case_1']['ece'] = _compute_ece(pred[case_1_idx], y[case_1_idx], conf_pred[case_1_idx])
  res['case_1']['brier'] = _compute_brier(conf[case_1_idx, :], y[case_1_idx])
  res['case_2']['count'] = np.sum(case_2_idx)
  res['case_2']['accuracy'] = (pred[case_2_idx] == y[case_2_idx]).mean()
  res['case_2']['ece'] = _compute_ece(pred[case_2_idx], y[case_2_idx], conf_pred[case_2_idx])
  res['case_2']['brier'] = _compute_brier(conf[case_2_idx, :], y[case_2_idx])
  res['case_3']['count'] = np.sum(case_3_idx)
  res['case_3']['accuracy'] = (pred[case_3_idx] == y[case_3_idx]).mean()
  res['case_3']['ece'] = _compute_ece(pred[case_3_idx], y[case_3_idx], conf_pred[case_3_idx])
  res['case_3']['brier'] = _compute_brier(conf[case_3_idx, :], y[case_3_idx])
  res['case_4']['count'] = np.sum(case_4_idx)
  res['case_4']['accuracy'] = (pred[case_4_idx] == y[case_4_idx]).mean()
  res['case_4']['ece'] = _compute_ece(pred[case_4_idx], y[case_4_idx], conf_pred[case_4_idx])
  res['case_4']['brier'] = _compute_brier(conf[case_4_idx, :], y[case_4_idx])

  return tran_type, tran_arg, res


def _convert_res_dict(ori_res_dict):
  res = {}
  for arg in ori_res_dict:
    if arg == 'acc' or arg == 'ece' or arg == 'cnt' or arg == 'brier':
      continue
    res[arg] = {}

    part_dict = ori_res_dict[arg]
    # ECE
    ece_dict = {x: part_dict[x]['ece'] for x in part_dict if x != 'acc' and x != 'ece' and x != 'cnt' and x != 'brier'}
    sorted_eces = sorted(ece_dict.items(), key=lambda x: x[1])

    for idx, case in enumerate(sorted_eces):
      if idx == 0:
        res[arg][case[0]] = {'ece': '\\textbf{{{:.6f}}}'.format(case[1])}
      elif idx == 3:
        res[arg][case[0]] = {'ece': '\\emph{{{:.6f}}}'.format(case[1])}
      else:
        res[arg][case[0]] = {'ece': '{:.6f}'.format(case[1])}
    # Accuracy
    ece_dict = {x: part_dict[x]['accuracy'] for x in part_dict if x != 'acc' and x != 'ece' and x != 'cnt' and x != 'brier'}
    sorted_eces = sorted(ece_dict.items(), key=lambda x: x[1], reverse=True)
    for idx, case in enumerate(sorted_eces):
      if idx == 0:
        res[arg][case[0]]['accuracy'] = '\\textbf{{{:.2f}}}'.format(case[1]*100.0)
      elif idx == 3:
        res[arg][case[0]]['accuracy'] = '\\emph{{{:.2f}}}'.format(case[1] * 100.0)
      else:
        res[arg][case[0]]['accuracy'] = '{:.2f}'.format(case[1]*100.0)

    # Brier
    brier_dict = {x: part_dict[x]['brier'] for x in part_dict if x != 'acc' and x != 'ece' and x != 'cnt' and x != 'brier'}
    sorted_briers = sorted(brier_dict.items(), key=lambda x: x[1])
    for idx, case in enumerate(sorted_briers):
      if idx == 0:
        res[arg][case[0]]['brier'] = '\\textbf{{{:.6f}}}'.format(case[1])
      elif idx == 3:
        res[arg][case[0]]['brier'] = '\\emph{{{:.6f}}}'.format(case[1])
      else:
        res[arg][case[0]]['brier'] = '{:.6f}'.format(case[1])

    # CNT
    ece_dict = {x: part_dict[x]['count'] for x in part_dict if x != 'acc' and x != 'ece' and x != 'cnt' and x != 'brier'}
    sorted_eces = sorted(ece_dict.items(), key=lambda x: x[1])
    for idx, case in enumerate(sorted_eces):
      res[arg][case[0]]['count'] = '{}'.format(case[1])

  return res


def _to_latex(model, trans, res_dict, metric):
  str_res_dict = _convert_res_dict(res_dict)

  content = '\\begin{table}[!hbt]\n'
  content += '\\caption{{{}}}\n'.format('Grouping Image Using {} transformation'.format('Zoom-Out' if trans == 'zoom' else 'Brightness'))
  content += '\\label{{tab:grp_img_{}_{}}}\n'.format(model, trans)
  content += '\\begin{center}\n'
  content += '\\resizebox{0.5\\textwidth}{!}{'
  content += '\\begin{tabular}{c|c|c|c||c|c}\n'
  content += '& &  \\multicolumn{{2}}{{c||}}{{{}}} & \\multicolumn{{2}}{{c}}{{Count}}\\\\\n'.format(metric.upper())
  content += '\\hline\n'
  content += '& Test Data & \\multicolumn{{2}}{{c||}}{{{:.6f}}} & \\multicolumn{{2}}{{c}}{{{}}}\\\\\n'.format(
      res_dict[metric], res_dict['cnt'])

  for idx, arg in enumerate(sorted(str_res_dict, reverse=True)):
    if idx == 0:
      content += '\\hline\n'
      content += '& ' + ('& Incr. & Not Incr.' * 2) + '\\\\\n'
    content += '\\hline\n'
    content += '\\multirow{{2}}{{*}}{{{}x}} & Change & {} & {} & {} & {}\\\\\n'.format(arg,
                                                                                       str_res_dict[arg]['case_1'][metric],
                                                                                       str_res_dict[arg]['case_2'][metric],
                                                                                       str_res_dict[arg]['case_1']['count'],
                                                                                       str_res_dict[arg]['case_2']['count'])

    content += '& No Change & {} & {} & {} & {}\\\\\n'.format(str_res_dict[arg]['case_3'][metric],
                                                              str_res_dict[arg]['case_4'][metric],
                                                              str_res_dict[arg]['case_3']['count'],
                                                              str_res_dict[arg]['case_4']['count'])

  content += '\\end{tabular}\n'
  content += '}\n'
  content += '\\end{center}\n'
  content += '\\end{table}\n'

  print(content)


def _to_plot(res_zoom, res_brightness, metric):
  def get_rank_dist(res):
    print(res)
    grp_rank_count = np.zeros((4, 4)) # group * rank

    cnt_args = 0
    for arg in res:
      if arg == 'acc' or arg == 'ece' or arg == 'cnt' or arg == 'brier':
        continue

      cnt_args += 1

      metric_dict = {int(case.split('_')[-1])-1: res[arg][case][metric]
                     for case in res[arg]
                     if case != 'acc' and case != 'ece' and case != 'cnt' and case != 'brier'}

      sorted_vals = sorted(metric_dict.items(), key=lambda x: x[1])
      sorted_groups = [x[0] for x in sorted_vals]
      for idx, grp in enumerate(sorted_groups):
        grp_rank_count[grp][idx] += 1

    grp_rank_count /= cnt_args
    print(grp_rank_count)
    cum_ratio = np.cumsum(grp_rank_count, axis=1)

    return grp_rank_count, cum_ratio

  grp_rank_count_zoom, cum_ratio_zoom = get_rank_dist(res_zoom)
  grp_rank_count_brightness, cum_ratio_brightness = get_rank_dist(res_brightness)

  fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
  width = 0.35
  ind = np.arange(4)*0.5
  ax1.bar(ind, grp_rank_count_zoom[:, 0], width, color='darkblue', edgecolor='k')
  ax1.bar(ind, grp_rank_count_zoom[:, 1], width, bottom=cum_ratio_zoom[:, 0], color='white', edgecolor='k')
  ax1.bar(ind, grp_rank_count_zoom[:, 2], width, bottom=cum_ratio_zoom[:, 1], color='tan', edgecolor='k')
  ax1.bar(ind, grp_rank_count_zoom[:, 3], width, bottom=cum_ratio_zoom[:, 2], color='firebrick', edgecolor='k')
  ax1.set_xlabel('Groups')
  ax1.set_xlim([-0.25, 1.75])
  ax1.set_xticks(ind)
  ax1.set_xticklabels(['G1', 'G2', 'G3', 'G4'])
  ax1.set_ylabel('Ratio')
  ax1.set_title('Zoom-Out\nTransformation', fontsize=12, fontweight='bold')

  ax2.bar(ind, grp_rank_count_brightness[:, 0], width, color='darkblue', edgecolor='k')
  ax2.bar(ind, grp_rank_count_brightness[:, 1], width, bottom=cum_ratio_brightness[:, 0], color='white', edgecolor='k')
  ax2.bar(ind, grp_rank_count_brightness[:, 2], width, bottom=cum_ratio_brightness[:, 1], color='tan', edgecolor='k')
  ax2.bar(ind, grp_rank_count_brightness[:, 3], width, bottom=cum_ratio_brightness[:, 2], color='firebrick', edgecolor='k')
  ax2.set_xlabel('Groups')
  ax2.set_xlim([-0.25, 1.75])
  ax2.set_xticks(ind)
  ax2.set_xticklabels(['G1', 'G2', 'G3', 'G4'])
  ax2.set_title('Brightness\nTransformation', fontsize=12, fontweight='bold')

  fig.legend(labels=['Rank 1', 'Rank 2', 'Rank 3', 'Rank 4'], fontsize='small', loc='upper right', bbox_to_anchor=(1, 0.9))

  # fig.tight_layout()
  plt.show()


def get_res(root_dir, trans_type, dataset, model, cal_method):
  res = {}
  for name in glob.glob(root_dir + trans_type + "_*"):
    print(name)
    tran_type, tran_arg, res_dir = analyze_dir(dataset, model, cal_method, name)

    if tran_arg == '0.05':
      continue

    res[tran_arg] = res_dir
    res['acc'] = res_dir['acc']
    res['ece'] = res_dir['ece']
    res['cnt'] = res_dir['cnt']
    res['brier'] = res_dir['brier']

  return res


def main():
  dataset = 'ImageNet'
  model = 'resnet152'
  root_dir = './outputs_onetime/{}/'.format(dataset)
  cal_method = 'ts'
  metric = 'ece'

  res_zoom = get_res(root_dir, 'zoom', dataset, model, cal_method)
  res_brightness = get_res(root_dir, 'brightness', dataset, model, cal_method)

  # print('%' + ('-' * 20) + (model + "," + trans_type) + ('-' * 20))
  # _to_latex(model, trans_type, res, metric)
  _to_plot(res_zoom, res_brightness, metric)


if __name__ == '__main__':
  main()
