import csv
import glob
import pandas as pd


def parse_name(filename):
  arr = filename[:-4].split('_')
  if len(arr) < 12:
    return None

  dataset = arr[1]
  model = arr[2]
  cal = arr[3]
  tran_name = arr[4]
  tran_min = arr[5]
  tran_max = arr[6]
  mode = arr[7]
  N = arr[9]
  L = arr[11]

  return [dataset, model, cal, tran_name, tran_min, tran_max, mode, N, L]


def analyze_val_file(name, eps):
  after_ece = None
  after_mce = None
  after_vce = None
  idx = None

  with open(name, 'r') as f:
    reader = csv.DictReader(f)
    for idx, line in enumerate(reader):
      prev_ece = float(line['trans_1_before_ece'])
      after_ece = float(line['trans_1_after_ece'])
      after_mce = float(line['trans_1_after_mce'])
      after_vce = float(line['trans_1_after_vce'])
      diff = abs(after_ece - prev_ece)

      if diff < eps:
        return idx+1, after_ece, after_mce, after_vce
      else:
        continue

  return idx, after_ece, after_mce, after_vce


def analyze_test_file(name, stop_idx):
  with open(name, 'r') as f:
    reader = csv.DictReader(f)

    for idx, line in enumerate(reader):
      if idx == 0:
        initial_ece = float(line['trans_1_before_ece'])
      if idx + 1 == stop_idx:
        final_ece = float(line['trans_1_after_ece'])
        final_mce = float(line['trans_1_after_mce'])
        final_vce = float(line['trans_1_after_vce'])
        return initial_ece, final_ece, final_mce, final_vce


def analyze_files(filenames, eps):
  print(analyze_val_file(filenames['val'], eps))
  stop_idx, val_ece, val_mce, val_vce = analyze_val_file(filenames['val'], eps)
  initial_ece, final_ece, final_mce, final_vce = analyze_test_file(filenames['test'], stop_idx)
  return stop_idx, val_ece, val_mce, val_vce, initial_ece, final_ece, final_mce, final_vce


def run():
  dataset = 'CIFAR100'
  root_dir = './summaries/' + dataset
  eps = 1e-2

  files = {}
  for name in glob.glob(root_dir + "/summaries*N_20*.csv"):
    parsed = parse_name(name)
    if parsed is None:
      continue

    key = tuple(parsed[:-3] + parsed[-2:])
    if key not in files:
      files[key] = {}

    files[key][parsed[-3]] = name
  columns = ['dataset', 'model', 'cal_method', 'tran_name', 'tran_min', 'tran_max', 'N', 'L', 'stop_idx', 'initial_ece', 'final_ece', 'final_mce', 'final_vce']
  df_res = pd.DataFrame(columns=columns)
  for fdict in files:
    print(fdict)
    stop_idx, val_ece, val_mce, val_vce, initial_ece, final_ece, final_mce, final_vce = analyze_files(files[fdict], eps)

    dict = {'dataset': fdict[0], 'model': fdict[1], 'cal_method': fdict[2], 'tran_name': fdict[3], 'tran_min': fdict[4],
            'tran_max': fdict[5], 'N': fdict[6], 'L': fdict[7], 'stop_idx': stop_idx,
            'val_ece': val_ece, 'val_mce': val_mce, 'val_vce': val_vce,
            'initial_ece': initial_ece, 'final_ece': final_ece, 'final_mce': final_mce, 'final_vce': final_vce}
    df_res = df_res.append(dict, ignore_index=True)

  df_res['increased'] = df_res['initial_ece'] < df_res['final_ece']
  print(df_res[['model', 'N', 'initial_ece', 'final_ece']])
  # df_res.to_csv('summaries/{}_summaries.csv'.format(dataset))


if __name__ == '__main__':
  run()
