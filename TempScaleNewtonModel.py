import numpy as np
import torch
import torch.nn.functional as F

from Ece import ECE


class TempScaleNewtonModel(torch.nn.Module):
  def __init__(self, model, device):
    super(TempScaleNewtonModel, self).__init__()
    self.model = model
    self.device = device
    self.temp = torch.nn.Parameter(torch.ones(1).to(self.device))

  def forward(self, input):
    logits = self.model(input)
    return self.temp_scale(logits)

  def temp_scale(self, logits):
    temp = self.temp.unsqueeze(1).expand(logits.size(0), logits.size(1))
    conf = F.softmax(logits, dim=1)
    conf = torch.clamp(conf, min=1e-16)
    return torch.log(conf) * temp

  def calibrate(self, loader):
    self.model.eval()

    conf_all = None
    true_all = None
    pred_all = None
    with torch.no_grad():
      for input, label in loader:
        input, label = input.to(self.device), label.to(self.device)

        logits = self.model(input)

        pred = logits.argmax(dim=1).detach().cpu().numpy()
        if pred_all is None:
          pred_all = pred
        else:
          pred_all = np.append(pred_all, pred.reshape((-1,)), axis=0)

        target = label.detach().cpu().numpy()
        if true_all is None:
          true_all = target
        else:
          true_all = np.append(true_all, target, axis=0)

        conf = F.softmax(logits, dim=1).detach().cpu().numpy()
        if conf_all is None:
          conf_all = conf
        else:
          conf_all = np.append(conf_all, conf, axis=0)

    self.calibrate_w_data(pred_all, true_all, conf_all, conf_all[np.arange(conf_all.shape[0]), pred_all])

  def calibrate_w_data(self, pred_all, true_all, conf_all, conf_pred_all):
    before_ece = ECE(15)
    before_ece.add_data(pred_all, true_all, conf_pred_all)
    print('Before ECE: {:.2f}, MCE: {:.2f}'.format(before_ece.compute_ECE() * 100.0, before_ece.compute_MCE() *100.0))

    cal = self._calibrate(conf_all, true_all)
    self.temp = torch.nn.Parameter(torch.ones(1).to(self.device) * cal)

    temp = self.temp.unsqueeze(1).expand(conf_all.shape[0], conf_all.shape[1])
    tc_conf_all = torch.Tensor(conf_all).to(self.device)
    tc_conf_all = torch.clamp(tc_conf_all, min=1e-16)
    scaled_conf_all = F.softmax(torch.log(tc_conf_all) * temp, dim=1).detach().cpu().numpy()
    after_ece = ECE(15)
    after_ece.add_data(pred_all, true_all, scaled_conf_all[np.arange(scaled_conf_all.shape[0]), pred_all])
    print('After ECE: {:.2f}, MCE: {:.2f}'.format(after_ece.compute_ECE() * 100.0, after_ece.compute_MCE() * 100.0))

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

  def _ece(self, logits, y_true):
    ece = ECE(15)
    y_pred = logits.argmax(dim=1)
    conf = F.softmax(logits, dim=1).gather(1, y_pred.view(-1, 1)).squeeze()

    y_pred = y_pred.cpu().numpy()
    y_true = y_true.cpu().numpy()
    conf = conf.detach().cpu().numpy()

    ece.add_data(y_pred, y_true, conf)

    return ece.compute_ECE()
