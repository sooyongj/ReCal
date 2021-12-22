import numpy as np
import os

from ReCal.TransformedRunner import TransformedRunner


class LogitsFileExport:
  def __init__(self, loader, loader_str, transformation_specs, n_trans, normalization_params, model, device, transformations):
    self.loader = loader
    self.loader_str = loader_str

    self.transformed_runner = TransformedRunner(self.loader, transformation_specs, n_trans, normalization_params, model, device, transformations)

  def export(self, output_dir):
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)

    fn_base = "{}_{}_{}_{}.npy"  # [F_TYPE]_[F_ARG]_[VAL_TEST]_[TYPE].npy

    for idx, (tran_type, tran_arg) in enumerate([('identity', 1.0)] + self.transformed_runner.transformations):
      print(idx + 1, tran_type, tran_arg)

      logits_all, y_all = self.transformed_runner.run_on_tran(tran_type, tran_arg)

      logits_fn = os.path.join(output_dir, fn_base.format(tran_type,
                                                          tran_arg,
                                                          self.loader_str,
                                                          "logits"))
      np.save(logits_fn, logits_all.numpy())
      print("\tstored {}.".format(logits_fn))

      if tran_type == 'identity':
        y_fn = os.path.join(output_dir, fn_base.format(tran_type,
                                                       tran_arg,
                                                       self.loader_str,
                                                       "ys"))
        np.save(y_fn, y_all.numpy())
        print("\tstored {}.".format(y_fn))



