import numpy as np
import random
import torch

from ReCal.TransformationsUtil import brightness, zoom_out, get_normalization_func


class TransformedRunner:
  def __init__(self, loader, transformation_specs, n_trans, normalization_params, model, device, transformations=None):
    self.loader = loader

    self.transformation_specs = transformation_specs

    self.normalization_params = normalization_params
    self.normalization_func = get_normalization_func(self.normalization_params)

    self.transformations = []
    self.n_trans = n_trans

    self.model = model
    self.device = device

    if transformations is None:
      self.transformations = self._sample_fs(self.transformation_specs)
    else:
      self.transformations = transformations
    print("Sampled Transformations:", self.transformations)

  def _get_trans(self, ftype, farg):
    if ftype == 'zoom':
      return lambda img: zoom_out(img, farg)
    elif ftype == 'brightness':
      return lambda img: brightness(img, farg)
    elif ftype == 'identity':
      return lambda img: img
    else:
      raise ValueError("Wrong Transformation type")

  def _sample_fs(self, allow_trans):
    possible = []
    for (trans_type, trans_min, trans_max) in allow_trans:
      for t_arg in np.arange(trans_min, trans_max, 0.01):
        possible += [(trans_type, round(t_arg, 2))]
    return sorted(random.sample(possible, self.n_trans))

  def run_on_tran(self, tran_type, tran_arg):
    self.model.eval()

    with torch.no_grad():
      logits_all = []

      y_all = []

      tran_func = self._get_trans(tran_type, tran_arg)

      for idx, (xs, ys) in enumerate(self.loader):
        xs, ys = xs.to(self.device), ys.to(self.device)

        xs = tran_func(xs)

        if self.normalization_func is not None:
          xs = self.normalization_func(xs)

        logits = self.model(xs)

        logits_all.append(logits.cpu())
        y_all.append(ys.cpu())

    logits_all = torch.cat(logits_all)
    y_all = torch.cat(y_all)

    cor = logits_all.argmax(dim=1) == y_all
    print("\tAccuracy: {}/{} = {:.2f} %".format(cor.sum(), y_all.shape[0],
                                                100.0 * cor.sum() / y_all.shape[0]))

    return logits_all, y_all

  def run_on_data(self, xs, tran_type, tran_arg):
    self.model.eval()

    with torch.no_grad():
      tran_func = self._get_trans(tran_type, tran_arg)
      xs = xs.to(self.device)

      xs = tran_func(xs)

      if self.normalization_func is not None:
        xs = self.normalization_func(xs)

      logits = self.model(xs).cpu()
    return logits
