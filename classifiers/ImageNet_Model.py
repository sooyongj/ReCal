import numpy as np
import torch
import torch.nn.functional as F
import torchvision

from TempScaleNewtonModel import TempScaleNewtonModel


class ImageNet_Model:
  def __init__(self, device, model_name):
    self.network = None
    self.is_train = False
    self.ts_network = None
    self.case_ts = None
    self.device = device
    self.model_name = model_name

  def init(self, is_train):
    self.is_train = is_train

    if self.is_train:
      pass
    else:
      if self.model_name == 'resnet152':
        self.network = torchvision.models.resnet152(pretrained=True, progress=False)
      elif self.model_name == 'densenet161':
        self.network = torchvision.models.densenet161(pretrained=True, progress=False)
      elif self.model_name == 'mobilenetv2':
        self.network = torchvision.models.mobilenet_v2(pretrained=True, progress=False)
      elif self.model_name == 'wrn101-2':
        self.network = torchvision.models.wide_resnet101_2(pretrained=True, progress=False)
      else:
        raise ValueError('Wrong model name', self.model_name)
      self.network = self.network.to(self.device)

  def train(self, loader, optimizer, epoch, log_interval=100):
    pass

  def test(self, network, loader, return_feature=False):
    network.eval()

    test_loss = 0
    correct = 0

    pred_all = None
    true_all = None
    conf_all = None
    feat_all = None

    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
      for data, target in loader:
        data, target = data.to(self.device), target.to(self.device)

        output = network(data)
        test_loss += criterion(output, target)
        pred = output.data.max(1, keepdim=True)[1]
        correct += (pred.eq(target.data.view_as(pred)).sum().item())

        conf = F.softmax(output, dim=1).detach().cpu().numpy()

        if pred_all is None:
          pred_all = pred.detach().cpu().numpy().reshape((-1,))
        else:
          pred_all = np.append(pred_all, pred.detach().cpu().numpy().reshape((-1,)), axis=0)
        if true_all is None:
          true_all = target.detach().cpu().numpy()
        else:
          true_all = np.append(true_all, target.detach().cpu(), axis=0)
        if conf_all is None:
          conf_all = conf
        else:
          conf_all = np.append(conf_all, conf, axis=0)
        if return_feature and feat_all is None:
          feat_all = data.detach().cpu().numpy()
        elif return_feature and feat_all is not None:
          feat_all = np.append(feat_all, data.detach().cpu().numpy(), axis=0)

    print('Test set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
      test_loss, correct, len(loader.dataset),
      (100. * correct / len(loader.dataset))))

    if return_feature:
      return feat_all, pred_all, true_all, conf_all, 100.0 * correct / len(loader.dataset)
    else:
      return None, pred_all, true_all, conf_all, 100.0 * correct / len(loader.dataset)

  def compute_output(self, network, loader):
    network.eval()

    test_loss = 0
    correct = 0

    pred_all = None
    true_all = None
    conf_all = None
    output_all = None

    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
      for data, target in loader:
        data, target = data.to(self.device), target.to(self.device)

        output = network(data)
        test_loss += criterion(output, target)
        pred = output.data.max(1, keepdim=True)[1]
        correct += (pred.eq(target.data.view_as(pred)).sum().item())

        conf = F.softmax(output, dim=1).detach().cpu().numpy()

        if pred_all is None:
          pred_all = pred.detach().cpu().numpy().reshape((-1,))
        else:
          pred_all = np.append(pred_all, pred.detach().cpu().numpy().reshape((-1,)), axis=0)
        if true_all is None:
          true_all = target.detach().cpu().numpy()
        else:
          true_all = np.append(true_all, target.detach().cpu(), axis=0)
        if conf_all is None:
          conf_all = conf
        else:
          conf_all = np.append(conf_all, conf, axis=0)
        if output_all is None:
          output_all = output.detach().cpu().numpy()
        else:
          output_all = np.append(output_all, output.detach().cpu().numpy(), axis=0)

    print('Test set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
      test_loss, correct, len(loader.dataset),
      (100. * correct / len(loader.dataset))))

    return output_all, pred_all, true_all, conf_all

  def test_case_ts(self, network, loader):
    network.eval()

    pred_all_case_1, conf_all_case_1, true_all_case_1 = [], [], []
    pred_all_case_2, conf_all_case_2, true_all_case_2 = [], [], []
    pred_all_case_3, conf_all_case_3, true_all_case_3 = [], [], []
    pred_all_case_4, conf_all_case_4, true_all_case_4 = [], [], []

    with torch.no_grad():
      for (xs, ys) in loader:
        xs, ys = xs.to(self.device), ys.to(self.device)

        output, case_1_idx, case_2_idx, case_3_idx, case_4_idx = network.forward_idx(xs)
        pred = output.data.max(1, keepdim=True)[1].squeeze()
        conf = F.softmax(output, dim=1)

        if case_1_idx.sum() > 0:
          pred_all_case_1.append(pred[case_1_idx])
          true_all_case_1.append(ys[case_1_idx])
          conf_all_case_1.append(conf[case_1_idx])

        if case_2_idx.sum() > 0:
          pred_all_case_2.append(pred[case_2_idx])
          true_all_case_2.append(ys[case_2_idx])
          conf_all_case_2.append(conf[case_2_idx])

        if case_3_idx.sum() > 0:
          pred_all_case_3.append(pred[case_3_idx])
          true_all_case_3.append(ys[case_3_idx])
          conf_all_case_3.append(conf[case_3_idx])

        if case_4_idx.sum() > 0:
          pred_all_case_4.append(pred[case_4_idx])
          true_all_case_4.append(ys[case_4_idx])
          conf_all_case_4.append(conf[case_4_idx])

      result = {'case_1': None, 'case_2': None, 'case_3': None, 'case_4': None}
      if len(pred_all_case_1) > 0:
        pred_all_case_1 = torch.cat(pred_all_case_1).detach().cpu().numpy()
        true_all_case_1 = torch.cat(true_all_case_1).detach().cpu().numpy()
        conf_all_case_1 = torch.cat(conf_all_case_1).detach().cpu().numpy()
        result['case_1'] = {'pred_all': pred_all_case_1, 'true_all': true_all_case_1, 'conf_all': conf_all_case_1}

      if len(pred_all_case_2) > 0:
        pred_all_case_2 = torch.cat(pred_all_case_2).detach().cpu().numpy()
        true_all_case_2 = torch.cat(true_all_case_2).detach().cpu().numpy()
        conf_all_case_2 = torch.cat(conf_all_case_2).detach().cpu().numpy()
        result['case_2'] = {'pred_all': pred_all_case_2, 'true_all': true_all_case_2, 'conf_all': conf_all_case_2}

      if len(pred_all_case_3) > 0:
        pred_all_case_3 = torch.cat(pred_all_case_3).detach().cpu().numpy()
        true_all_case_3 = torch.cat(true_all_case_3).detach().cpu().numpy()
        conf_all_case_3 = torch.cat(conf_all_case_3).detach().cpu().numpy()
        result['case_3'] = {'pred_all': pred_all_case_3, 'true_all': true_all_case_3, 'conf_all': conf_all_case_3}

      if len(pred_all_case_4) > 0:
        pred_all_case_4 = torch.cat(pred_all_case_4).detach().cpu().numpy()
        true_all_case_4 = torch.cat(true_all_case_4).detach().cpu().numpy()
        conf_all_case_4 = torch.cat(conf_all_case_4).detach().cpu().numpy()
        result['case_4'] = {'pred_all': pred_all_case_4, 'true_all': true_all_case_4, 'conf_all': conf_all_case_4}

    return result

  def run_temp_scaling(self, val_loader):
    # self.ts_network = TempScalingModel(self.network)
    self.ts_network = TempScaleNewtonModel(self.network, device=self.device)
    self.ts_network.calibrate(val_loader)
    # if self.is_train:
    #   self.ts_network.calibrate(val_loader)
    #   self.ts_network.save('model_ts_imagenet_resnet152.pt')
    # else:
    #   self.ts_network.load('model_ts_imagenet_resnet152.pt')

    self.ts_network = self.ts_network.to(self.device)