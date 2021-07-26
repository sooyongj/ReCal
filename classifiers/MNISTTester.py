import torch
import torch.nn.functional as F

import numpy as np


def test(network, loader, return_feature=False):
  network.eval()
  test_loss = 0
  correct = 0

  pred_all = None
  true_all = None
  conf_all = None
  feat_all = None

  with torch.no_grad():
    for data, target in loader:
      if torch.cuda.is_available():
        data, target = data.cuda(), target.cuda()

      output = network(data)
      test_loss += F.nll_loss(output, target, size_average=False).item()
      pred = output.data.max(1, keepdim=True)[1]
      correct += (pred.eq(target.data.view_as(pred)).sum().item())

      conf = F.softmax(output, dim=1).cpu().numpy()

      if pred_all is None:
        pred_all = pred.numpy().reshape((-1,))
      else:
        pred_all = np.append(pred_all, pred.numpy().reshape((-1,)), axis=0)
      if true_all is None:
        true_all = target.numpy()
      else:
        true_all = np.append(true_all, target, axis=0)
      if conf_all is None:
        conf_all = conf
      else:
        conf_all = np.append(conf_all, conf, axis=0)
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