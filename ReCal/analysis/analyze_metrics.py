import numpy as np
import glob
import os
from scipy.special import softmax
from sklearn.metrics import log_loss
import pandas as pd

from ReCal.Ece import ECE


def get_values(logits_fn, y_fn):
  logits = np.load(logits_fn)
  y = np.load(y_fn)

  pred = np.argmax(logits, axis=1)

  conf = softmax(logits, axis=1)
  conf_pred = conf[np.arange(y.size), pred]

  return logits, y, pred, conf, conf_pred


def compute_brier(conf, y):
  targets = np.zeros_like(conf)
  targets[np.arange(y.size), y] = 1

  return np.average(np.sum((conf - targets) ** 2, axis=1)) / conf.shape[1] #??


def compute_ece(pred, y, conf_pred):
  ece = ECE(15)
  ece.add_data(pred, y, conf_pred)

  return ece.compute_ECE()


def compute_cwece(pred, y, conf):
  n_classes = conf.shape[1]
  final_ece = 0

  oldy = y
  y = np.zeros_like(conf)
  y[np.arange(pred.size), oldy] = 1

  def bin_func(p, y, idx):
      return (np.abs(np.mean(p[idx] - np.mean(y[idx, k])))) * np.sum(idx)/ len(probs)

  ece = 0
  for k in range(n_classes):
      probs = conf[:,k]

      idx = np.digitize(probs, np.linspace(0, 1, 15), right=True)-1
#       bin_func = lambda p, y, idx: (np.abs(np.mean(p[idx] - np.mean(y[idx, k])))) * np.sum(idx)/ len(probs)
      for i in np.unique(idx):
          ece += bin_func(conf[:,k], y, idx == i)
  return ece


def analyze_cal(logits_fn, y_fn, dataset, model, cal_method):
  logits, y, pred, conf, conf_pred = get_values(logits_fn, y_fn)

  error = (y != pred).mean() * 100.0
  brier_score = compute_brier(conf, y)
  ece_score = compute_ece(pred, y, conf_pred)
  nll_score = log_loss(y_true=y, y_pred=conf)
  cwece_score = compute_cwece(pred, y, conf)

  res_dict = {'Dataset': dataset, 'Model': model, 'Calibration': cal_method,
              'Error': error, 'ECE': ece_score, 'ECE-CW': cwece_score, 'Brier': brier_score, 'NLL': nll_score}
  return res_dict


def load_odir_res(root_dir):
  res = []
  for fn in glob.glob(os.path.join(root_dir, '*.p')):
    df_res = pd.read_pickle(fn)

    if isinstance(df_res, tuple):
      cols = ["Error_test", "ECE_test", "ECE2_test", "ECE_CW_test", "ECE_CW2_test", "ECE_FULL_test", "ECE_FULL2_test",
              "MCE_test", "MCE2_test", "Loss_test", "Brier_test"]
      cols_ens = [col + "_ens" for col in cols]
      df_ens = df_res[1].loc[:, cols]
      df_ens.columns = cols_ens
      df_merged = pd.concat([df_res[0], df_ens], axis=1)
      df_sorted_res_merged = df_merged.sort_values(by='Loss')

      name = df_sorted_res_merged.iloc[0]['Name']
      name_arr = name.split('_')
      if len(name_arr) < 3:
        print('ignored', name, fn)
        continue

      dataset = name_arr[0]
      model = name_arr[1]
      cal_method = '_'.join(name_arr[2:])

      l2 = df_sorted_res_merged.iloc[0]['L2']
      mu = df_sorted_res_merged.iloc[0]['mu']

      print(dataset, l2, mu)
      print(df_sorted_res_merged.iloc[0])

      elem_dict = {'Dataset': dataset, "Model": model, "Calibration": cal_method,
                   'Error': df_sorted_res_merged.iloc[0]['Error_test_ens'],
                   'ECE': df_sorted_res_merged.iloc[0]['ECE_test_ens'],
                   'Brier': df_sorted_res_merged.iloc[0]['Brier_test_ens'],
                   'NLL': df_sorted_res_merged.iloc[0]['Loss_test_ens'],
                   'ECE-CW': df_sorted_res_merged.iloc[0]['ECE_CW_test_ens']}
      res.append(elem_dict)
    else:
      # Vector Scaling
      for idx, row in df_res.iterrows():
        name = row.Name
        if '_calib' not in name or '_val_calib' in name: # uncalibrated or validation set
          continue
        name_arr = name.split('_')
        dataset = name_arr[0]
        model = name_arr[1]
        elem_dict = {'Dataset': dataset, 'Model': model, 'Calibration': 'VecScaling',
                     'Error': row['Error'],
                     'ECE': row['ECE'],
                     'Brier': row['Brier'],
                     'NLL': row['Loss'],
                     'ECE-CW': row['ECE_CW']}
        res.append(elem_dict)
  res = pd.DataFrame(res)
  return res


def run():
  odir_result_path = './odir_results'
  df_res_odir = load_odir_res(odir_result_path)

  cols = ['Dataset', 'Model', 'Calibration', 'Error', 'ECE', 'Brier', 'NLL', 'ECE-CW']
  df_res = pd.DataFrame(columns=cols)
  df_res = df_res.append(df_res_odir)
  dataset_models = {
          'CIFAR10': ['densenet40', 'lenet5', 'resnet110', 'resnet110sd', 'wrn28-10'],
          'CIFAR100': ['densenet40', 'lenet5', 'resnet110', 'resnet110sd', 'wrn28-10'],
          'ImageNet': ['densenet161', 'resnet152'],
          'SVHN': ['resnet152sd']
          }

  for dataset in dataset_models:
    print('Dataset: {}'.format(dataset))
    for model in dataset_models[dataset]:
      print(model)
      # uncalibrated
      logits_fn = 'outputs/{}/{}_zoom_0.1_0.9_20/uncalibrated/{}_{}_type_identity_arg_1.0_test_logits.npy'.format(dataset, model, dataset, model)
      y_fn = 'outputs/{}/{}_zoom_0.1_0.9_20/uncalibrated/{}_{}_type_identity_arg_1.0_test_y.npy'.format(dataset, model, dataset, model)

      if not os.path.exists(logits_fn) or not os.path.exists(y_fn):
        continue

      uncal_dict = analyze_cal(logits_fn, y_fn, dataset, model, 'uncalibrated')
      df_res = df_res.append(uncal_dict, ignore_index=True)
      
      # ts
      logits_fn = 'outputs/{}/{}_zoom_0.1_0.9_20/ts/{}_{}_type_identity_arg_1.0_test_logits.npy'.format(dataset, model, dataset, model)
      y_fn = 'outputs/{}/{}_zoom_0.1_0.9_20/ts/{}_{}_type_identity_arg_1.0_test_y.npy'.format(dataset, model, dataset, model)

      if not os.path.exists(logits_fn) or not os.path.exists(y_fn):
        continue

      ts_dict = analyze_cal(logits_fn, y_fn, dataset, model, 'ts')
      df_res = df_res.append(ts_dict, ignore_index=True)

      # # ours_old - Z (0.5, 0.9, 10)
      # logits_fn = 'outputs/{}/ours_old/zoom_0.5_0.9_N_10/uncalibrated/{}_{}_type_identity_arg_1.0_test_logits.npy'.format(dataset, dataset, model)
      # y_fn = 'outputs/{}/ours_old/zoom_0.5_0.9_N_10/uncalibrated/{}_{}_type_identity_arg_1.0_test_y.npy'.format(dataset, dataset, model)
      #
      # if not os.path.exists(logits_fn) or not os.path.exists(y_fn):
      #   continue
      #
      # our_dict = analyze_cal(logits_fn, y_fn, dataset, model, 'ours_old_z.5.910')
      # df_res = df_res.append(our_dict, ignore_index=True)
      #
      # # ours_old - Z (0.1, 0.9, 20)
      # logits_fn = 'outputs/{}/ours_old/zoom_0.1_0.9_N_20/uncalibrated/{}_{}_type_identity_arg_1.0_test_logits.npy'.format(dataset, dataset, model)
      # y_fn = 'outputs/{}/ours_old/zoom_0.1_0.9_N_20/uncalibrated/{}_{}_type_identity_arg_1.0_test_y.npy'.format(dataset, dataset, model)
      #
      # if not os.path.exists(logits_fn) or not os.path.exists(y_fn):
      #   continue
      #
      # our_dict = analyze_cal(logits_fn, y_fn, dataset, model, 'ours_old_z.1.920')
      # df_res = df_res.append(our_dict, ignore_index=True)
      #
      # # ours_old - B (0.1, 0.9, 20)
      # logits_fn = 'outputs/{}/ours_old/brightness_0.1_0.9_N_20/uncalibrated/{}_{}_type_identity_arg_1.0_test_logits.npy'.format(
      #   dataset, dataset, model)
      # y_fn = 'outputs/{}/ours_old/brightness_0.1_0.9_N_20/uncalibrated/{}_{}_type_identity_arg_1.0_test_y.npy'.format(dataset,
      #                                                                                                             dataset,
      #                                                                                                             model)
      # if not os.path.exists(logits_fn) or not os.path.exists(y_fn):
      #   continue
      #
      # our_dict = analyze_cal(logits_fn, y_fn, dataset, model, 'ours_old_b.1.920')
      # df_res = df_res.append(our_dict, ignore_index=True)

      # ours_new - Z (0.1, 0.9, 20)
      logits_fn = 'outputs/{}/ours/zoom_0.1_0.9_N_20/uncalibrated/{}_{}_type_identity_arg_1.0_test_logits.npy'.format(
        dataset, dataset, model)
      y_fn = 'outputs/{}/ours/zoom_0.1_0.9_N_20/uncalibrated/{}_{}_type_identity_arg_1.0_test_y.npy'.format(dataset,
                                                                                                            dataset,
                                                                                                            model)
      if not os.path.exists(logits_fn) or not os.path.exists(y_fn):
        continue

      our_dict = analyze_cal(logits_fn, y_fn, dataset, model, 'ours_new_z.1.920')
      df_res = df_res.append(our_dict, ignore_index=True)

      # # ours_new - Z (0.1, 0.9, 20) - ts
      # logits_fn = 'outputs/{}/ours/zoom_0.1_0.9_N_20/ts/{}_{}_type_identity_arg_1.0_test_logits.npy'.format(
      #   dataset, dataset, model)
      # y_fn = 'outputs/{}/ours/zoom_0.1_0.9_N_20/ts/{}_{}_type_identity_arg_1.0_test_y.npy'.format(dataset,
      #                                                                                             dataset,
      #                                                                                             model)
      # if not os.path.exists(logits_fn) or not os.path.exists(y_fn):
      #   continue
      #
      # our_dict = analyze_cal(logits_fn, y_fn, dataset, model, 'ours_new_z_ts.1.920')
      # df_res = df_res.append(our_dict, ignore_index=True)

      # ours_new - Z (0.5, 0.9, 10)
      logits_fn = 'outputs/{}/ours/zoom_0.5_0.9_N_10/uncalibrated/{}_{}_type_identity_arg_1.0_test_logits.npy'.format(
        dataset, dataset, model)
      y_fn = 'outputs/{}/ours/zoom_0.5_0.9_N_10/uncalibrated/{}_{}_type_identity_arg_1.0_test_y.npy'.format(dataset,
                                                                                                            dataset,
                                                                                                            model)
      if not os.path.exists(logits_fn) or not os.path.exists(y_fn):
        continue

      our_dict = analyze_cal(logits_fn, y_fn, dataset, model, 'ours_new_z.5.910')
      df_res = df_res.append(our_dict, ignore_index=True)

      # # ours_new - Z (0.5, 0.9, 10) - ts
      # logits_fn = 'outputs/{}/ours/zoom_0.5_0.9_N_10/ts/{}_{}_type_identity_arg_1.0_test_logits.npy'.format(
      #   dataset, dataset, model)
      # y_fn = 'outputs/{}/ours/zoom_0.5_0.9_N_10/ts/{}_{}_type_identity_arg_1.0_test_y.npy'.format(dataset,
      #                                                                                             dataset,
      #                                                                                             model)
      # if not os.path.exists(logits_fn) or not os.path.exists(y_fn):
      #   continue
      #
      # our_dict = analyze_cal(logits_fn, y_fn, dataset, model, 'ours_new_z_ts.5.910')
      # df_res = df_res.append(our_dict, ignore_index=True)

      # ours_new - b (0.1, 0.9, 20)
      logits_fn = 'outputs/{}/ours/brightness_0.1_0.9_N_20/uncalibrated/{}_{}_type_identity_arg_1.0_test_logits.npy'.format(
        dataset, dataset, model)
      y_fn = 'outputs/{}/ours/brightness_0.1_0.9_N_20/uncalibrated/{}_{}_type_identity_arg_1.0_test_y.npy'.format(dataset,
                                                                                                  dataset,
                                                                                                  model)

      if not os.path.exists(logits_fn) or not os.path.exists(y_fn):
        continue

      our_dict = analyze_cal(logits_fn, y_fn, dataset, model, 'ours_new_b.1.920')
      df_res = df_res.append(our_dict, ignore_index=True)

      # # ours_new - b (0.1, 0.9, 20) -ts
      # logits_fn = 'outputs/{}/ours/brightness_0.1_0.9_N_20/ts/{}_{}_type_identity_arg_1.0_test_logits.npy'.format(
      #   dataset, dataset, model)
      # y_fn = 'outputs/{}/ours/brightness_0.1_0.9_N_20/ts/{}_{}_type_identity_arg_1.0_test_y.npy'.format(dataset,
      #                                                                                             dataset,
      #                                                                                             model)
      #
      # if not os.path.exists(logits_fn) or not os.path.exists(y_fn):
      #   continue
      #
      # our_dict = analyze_cal(logits_fn, y_fn, dataset, model, 'ours_new_b_ts.1.920')
      # df_res = df_res.append(our_dict, ignore_index=True)

  df_res = df_res.sort_values(by=['Dataset', 'Model', 'Calibration'])

  pd.set_option('display.max_columns', None)

  # print(df_res.set_index(['Dataset', 'Model', 'Calibration']))

  convert_to_latex_table(df_res)


def _to_latex(df_result, metric):
  alg_order = ['uncalibrated', 'ts', 'VecScaling', 'ms_odir', 'dir_odir']
  # alg_order += ['ours_old_z.1.920', 'ours_old_z.5.910', 'ours_old_b.1.920']
  alg_order += ['ours_new_z.1.920']
  # alg_order += ['ours_new_z_ts.1.920']
  alg_order += ['ours_new_z.5.910']
  # alg_order += ['ours_new_z_ts.5.910']
  alg_order += ['ours_new_b.1.920']
  # alg_order += ['ours_new_b_ts.1.920']
  model_name_map = {'densenet40': 'DenseNet40', 'lenet5': 'LeNet5',
                    'resnet110': 'ResNet110', 'resnet110sd': 'ResNet110 SD',
                    'wrn28-10': 'WRN 28-10',
                    'densenet161': 'DenseNet161', 'resnet152': 'ResNet152',
                    'resnet152sd': 'ResNet152 SD',
                    'mobilenetv2': 'MobileNet V2', 'wrn101-2': 'WRN 101-2'}

  datasets = df_result.Dataset.unique()
  content = '\\begin{table*}[!hbt]\n'
  content += '\\caption{{{}}}\n'.format(metric)
  content += '\\label{{tab:exp_res_{}}}\n'.format(metric)
  content += '\\begin{center}\n'
  content += '\\resizebox{\\textwidth}{!}{'
  content += '\\begin{tabular}{c|c' + ('|c' * len(alg_order)) + '}\n'
  content += 'Dataset & Model & Uncal. & TS & VS & MS-ODIR & Dir-ODIR'
  # content += ' & \\thead{Ours-old\\\\(z, .1-.9, 20)} & \\thead{Ours-old\\\\(z, .5-.9, 10)} & \\thead{Ours-old\\\\(b, .1-.9, 20)}'
  content += ' & \\thead{Ours-new\\\\(z, .1-.9, 20)}'
  # content += ' & \\thead{Ours-new\\\\(z, .1-.9, 20), ts}'
  content += ' & \\thead{Ours-new\\\\(z, .5-.9, 10)}'
  # content += ' & \\thead{Ours-new\\\\(z, .5-.9, 10, ts)}'
  content += ' & \\thead{Ours-new\\\\(b, .1-.9, 20)}'
  # content += ' & \\thead{Ours-new\\\\(b, .1-.9, 20, ts)}'
  content += '\\\\\n'
  content += '\\hline\n\\hline\n'

  # ranks = {c: [] for c in alg_order}
  for dataset in datasets:
    models = df_result[df_result.Dataset == dataset].Model.unique()
    for model in models:
      temp_res = {c: '' for c in alg_order}
      df_extract = df_result[(df_result.Dataset == dataset) & (df_result.Model == model)]

      for _, row in df_extract.iterrows():
        temp_res[row["Calibration"]] = '{:.2f}'.format(row[metric]) if metric == 'Error' else '{:.6f}'.format(row[metric])
      df_sorted = df_extract.sort_values(by=metric)
      sorted_values = df_sorted[metric].unique().tolist()
      items_best = df_extract[df_extract[metric] == sorted_values[0]].Calibration.unique().tolist()
      for item in items_best:
        temp_res[item] = '\\textbf{{{}}}'.format(temp_res[item])
      if len(sorted_values) > 1:
        items_2nd_best = df_extract[df_extract[metric] == sorted_values[1]].Calibration.unique().tolist()
        for item in items_2nd_best:
          temp_res[item] = '\\underline{{{}}}'.format(temp_res[item])

      # rank = 1
      # tie_algs = []
      # tie_rank = []
      # prev = None
      # for _, row in df_sorted.iterrows():
      #   v = row[metric]
      #
      #   if prev is not None and prev != v:
      #     avg_tie_rank = np.average(np.array(tie_rank))
      #     for c in tie_algs:
      #       ranks[c].append(avg_tie_rank)
      #     tie_rank = []
      #     tie_algs = []
      #
      #   tie_algs.append(row['Calibration'])
      #   tie_rank.append(rank)
      #
      #   prev = v
      #   rank += 1
      #
      # avg_tie_rank = np.average(np.array(tie_rank))
      # for c in tie_algs:
      #   ranks[c].append(avg_tie_rank)

      arr = [dataset, model_name_map[model]]
      arr += [temp_res[c] for c in alg_order]

      content += ' & '.join(arr) + '\\\\\n'
    content += '\\hline\n'

  # avg_ranks = ['{:.2f}'.format(np.mean(np.array(ranks[cal]))) for cal in alg_order]
  # content += '\\multicolumn{2}{c|}{{Avg. Rank}} & ' + ' & '.join(avg_ranks) + '\\\\\n'

  content += '\\end{tabular}\n'
  content += '}\n'
  content += '\\end{center}\n'
  content += '\\end{table*}'

  return content


def convert_to_latex_table(df_result):
  metrics = ['ECE', 'Brier', 'Error', 'NLL', 'ECE-CW']

  for metric in metrics:
    print('%' + ('-' * 30) + metric + ('-'* 30))
    content = _to_latex(df_result, metric)
    print(content)


if __name__ == '__main__':
  run()
