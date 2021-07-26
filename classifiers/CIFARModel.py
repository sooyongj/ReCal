import numpy as np
import torch
import torch.nn.functional as F

from pytorch_classification.models.cifar import resnet, wrn

from TempScaleNewtonModel import TempScaleNewtonModel
from model.CIFARLeNet5 import LeNet5
from model.densenet import DenseNet3
from model.resnet import resnetsd


class CIFARModel:
  def __init__(self, device, model_name, is_ten=True):
    self.network = None
    self.is_train = False
    self.ts_network = None
    self.case_ts = None
    self.n_class = 10 if is_ten else 100
    self.device = device
    self.model_name = model_name
    self.is_ten = is_ten

  def init(self, is_train):
    self.is_train = is_train
    if self.model_name == 'lenet5':
      self.network = LeNet5(n_labels=self.n_class)
    elif self.model_name == 'densenet40':
      self.network = DenseNet3(40, self.n_class, reduction=1.0, bottleneck=False, dropRate=0)
    elif self.model_name == 'resnet110':
      depth = 110
      self.network = resnet(num_classes=self.n_class,
                            depth=depth,
                            block_name='BasicBlock')
    elif self.model_name == 'resnet110sd':
      layers = 110
      self.network = resnetsd(layers=layers, prob=0.5, num_classes=self.n_class)
    elif self.model_name == 'wrn28-10':
      self.network = wrn(num_classes=self.n_class,
                         depth=28,
                         widen_factor=10,
                         dropRate=0)
    else:
      raise ValueError("Not supported Model: {}".format(self.model_name))

    self.network = self.network.to(self.device)

    if self.is_train:
      pass
    else:
      chkptname = './checkpoint/cifar{}_{}.pth'.format(self.n_class, self.model_name)

      if self.device.type == 'cuda':
        checkpoint = torch.load(chkptname)
      else:
        checkpoint = torch.load(chkptname, map_location=torch.device('cpu'))

      self.network.load_state_dict(checkpoint['state_dict'])
      acc = checkpoint['acc']
      print('stored acc: {:.2f} %'.format(acc * 100.0))
      return checkpoint['val_idx']

  def train(self, loader, optimizer, epoch, log_interval=100):
    # TODO: Modify this!
    print('\nEpoch: %d' % epoch)
    self.network.train()
    train_loss = 0
    correct = 0
    total = 0
    criterion = torch.nn.CrossEntropyLoss()
    for batch_idx, (inputs, targets) in enumerate(loader):
      inputs, targets = inputs.to(self.device), targets.to(self.device)

      optimizer.zero_grad()
      outputs = self.network(inputs)
      loss = criterion(outputs, targets)
      loss.backward()
      optimizer.step()

      train_loss += loss.item()
      _, predicted = outputs.max(1)
      total += targets.size(0)
      correct += predicted.eq(targets).sum().item()

      # progress_bar(batch_idx, len(loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
      #              % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
      if batch_idx % log_interval == 0:
        print(batch_idx, len(loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
              % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

  def test(self, network, loader, return_feature=False):
    network.eval()
    test_loss = 0
    correct = 0

    pred_all = np.zeros((len(loader.dataset),), dtype=np.int)
    true_all = np.zeros((len(loader.dataset),), dtype=np.int)
    conf_all = np.zeros((len(loader.dataset), self.n_class), dtype=np.float32)
    feat_all = None

    criterion = torch.nn.CrossEntropyLoss()
    start_idx = 0
    with torch.no_grad():
      for data, target in loader:
        data, target = data.to(self.device), target.to(self.device)

        output = network(data)
        test_loss += criterion(output, target)
        pred = output.data.max(1, keepdim=True)[1]
        correct += (pred.eq(target.data.view_as(pred)).sum().item())

        conf = F.softmax(output, dim=1).cpu().numpy()

        end_idx = start_idx + target.size(0)

        pred_all[start_idx:end_idx] = pred.cpu().numpy().reshape((-1,))
        true_all[start_idx:end_idx] = target.cpu().numpy()
        conf_all[start_idx:end_idx, :] = conf

        start_idx += target.size(0)

        if return_feature and feat_all is None:
          feat_all = data.cpu().numpy()
        elif return_feature and feat_all is not None:
          feat_all = np.append(feat_all, data.cpu().numpy(), axis=0)

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

    pred_all = np.zeros((len(loader.dataset),), dtype=np.int)
    true_all = np.zeros((len(loader.dataset),), dtype=np.int)
    conf_all = np.zeros((len(loader.dataset), self.n_class), dtype=np.float32)
    output_all = np.zeros((len(loader.dataset), self.n_class), dtype=np.float32)

    criterion = torch.nn.CrossEntropyLoss()
    start_idx = 0
    with torch.no_grad():
      for data, target in loader:
        data, target = data.to(self.device), target.to(self.device)

        output = network(data)
        test_loss += criterion(output, target)
        pred = output.data.max(1, keepdim=True)[1]
        correct += (pred.eq(target.data.view_as(pred)).sum().item())

        conf = F.softmax(output, dim=1).cpu().numpy()

        end_idx = start_idx + target.size(0)

        pred_all[start_idx:end_idx] = pred.cpu().numpy().reshape((-1,))
        true_all[start_idx:end_idx] = target.cpu().numpy()
        conf_all[start_idx:end_idx, :] = conf
        output_all[start_idx:end_idx, :] = output.cpu().numpy()

        start_idx += target.size(0)

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
    self.ts_network = TempScaleNewtonModel(self.network, device=self.device)
    self.ts_network.calibrate(val_loader)
    # self.ts_network = TempScalingModel(self.network)
    # self.ts_network.calibrate(val_loader)
    # if self.is_train:
    #   self.ts_network.calibrate(val_loader)
    #   self.ts_network.save('model_ts_cifar_resnet152.pt')
    # else:
    #   self.ts_network.load('model_ts_cifar_resnet152.pt')

    self.ts_network = self.ts_network.to(self.device)
