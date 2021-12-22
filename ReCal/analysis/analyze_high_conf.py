import numpy as np
from scipy.special import softmax

from ReCal.Ece import ECE


def get_c99_cnt_acc(logits_fn, y_fn):
  logits = np.load(logits_fn)
  y = np.load(y_fn)

  pred = logits.argmax(axis=1)
  conf = softmax(logits, axis=1)
  conf_pred = conf[np.arange(pred.shape[0]), pred]

  tot_acc = (pred == y).mean()

  c99_index = conf_pred >= 0.99
  c99_cnt = np.sum(c99_index)
  c99_acc = (pred[c99_index] == y[c99_index]).mean()
  c99_avg_ece = conf_pred[c99_index].mean()

  ece = ECE(15)
  ece.add_data(pred, y, conf_pred)

  return tot_acc, conf_pred, ece, c99_cnt, c99_acc, c99_avg_ece


def run():
  map = {'CIFAR10': ['densenet40', 'lenet5', 'resnet110', 'resnet110sd', 'wrn28-10'],
         'CIFAR100': ['densenet40', 'lenet5', 'resnet110', 'resnet110sd', 'wrn28-10'],
         'ImageNet': ['densenet161', 'resnet152'],
         'SVHN': ['resnet152sd']}

  for dataset in map:
    for model in map[dataset]:
      print(dataset, model)

      # uncal
      logits_fn = './outputs/{d}/{m}_zoom_0.1_0.9_20/uncalibrated/{d}_{m}_type_identity_arg_1.0_test_logits.npy'.format(d=dataset, m=model)
      y_fn = './outputs/{d}/{m}_zoom_0.1_0.9_20/uncalibrated/{d}_{m}_type_identity_arg_1.0_test_y.npy'.format(d=dataset, m=model)

      tot_acc, tot_conf_pred_uncal, ece_uncal, c99_cnt, c99_acc, c99_avg_ece = get_c99_cnt_acc(logits_fn, y_fn)
      print('[Uncal] Total Accuracy: {:.2f}, C99 Count: {}, C99 Accuracy: {:.2f}, C99 Avg. ECE: {:.6f}, diff: {:.6f}'.format(tot_acc * 100, c99_cnt, c99_acc * 100, c99_avg_ece, abs(c99_acc - c99_avg_ece)))
      print()

      # TS
      logits_fn = './outputs/{d}/{m}_zoom_0.1_0.9_20/ts/{d}_{m}_type_identity_arg_1.0_test_logits.npy'.format(d=dataset, m=model)
      y_fn = './outputs/{d}/{m}_zoom_0.1_0.9_20/ts/{d}_{m}_type_identity_arg_1.0_test_y.npy'.format(d=dataset, m=model)

      tot_acc, tot_conf_pred_ts, ece_ts, c99_cnt, c99_acc, c99_avg_ece = get_c99_cnt_acc(logits_fn, y_fn)
      print('[TS] Total Accuracy: {:.2f}, C99 Count: {}, C99 Accuracy: {:.2f}, C99 Avg. ECE: {:.6f}, diff: {:.6f}'.format(tot_acc * 100, c99_cnt, c99_acc * 100, c99_avg_ece, abs(c99_acc - c99_avg_ece)))

      # OURS
      logits_fn = './outputs/{d}/ours/zoom_0.1_0.9_N_20/uncalibrated/{d}_{m}_type_identity_arg_1.0_test_logits.npy'.format(d=dataset, m=model)
      y_fn = './outputs/{d}/ours/zoom_0.1_0.9_N_20/uncalibrated/{d}_{m}_type_identity_arg_1.0_test_y.npy'.format(d=dataset, m=model)

      tot_acc, tot_conf_pred_ours, ece_ours, c99_cnt, c99_acc, c99_avg_ece = get_c99_cnt_acc(logits_fn, y_fn)
      print('[Ours] Total Accuracy: {:.2f}, C99 Count: {}, C99 Accuracy: {:.2f}, C99 Avg. ECE: {:.6f}, diff: {:.6f}'.format(tot_acc * 100, c99_cnt, c99_acc * 100, c99_avg_ece, abs(c99_acc - c99_avg_ece)))


  # import matplotlib.pyplot as plt
  # plt.figure()
  # bins=np.arange(0, 1.01, 0.01)
  # uncal, base_uncal = np.histogram(tot_conf_pred_uncal, bins=bins)
  # uncal = np.cumsum(uncal)
  # uncal = uncal / uncal[-1]
  # ts, base_ts = np.histogram(tot_conf_pred_ts, bins=bins)
  # ts = np.cumsum(ts)
  # ts = ts / ts[-1]
  # ours, base_ours = np.histogram(tot_conf_pred_ours, bins=bins)
  # ours = np.cumsum(ours)
  # ours = ours / ours[-1]
  # plt.plot(bins[:-1], uncal)
  # plt.plot(bins[:-1], ts)
  # plt.plot(bins[:-1], ours)
  # plt.legend(['Uncal', 'TS', 'Ours'])
  # plt.show()

  # import matplotlib.pyplot as plt
  # def draw_rd(ece):
  #   plt.figure()
  #   ece_mat = ece.get_ECE_mat()
  #   plt.bar(x=ece.bin_lowers, height=ece_mat[:, 1], width=1/len(ece.bin_lowers), align='edge', edgecolor='k')
  #   # plt.plot(ece_uncal.bin_uppers, ece_mat[:,1])
  #   plt.plot([0,1], [0,1], 'g')
  #
  # draw_rd(ece_uncal)
  # draw_rd(ece_ts)
  # draw_rd(ece_ours)
  # plt.show()


if __name__ == '__main__':
  run()
