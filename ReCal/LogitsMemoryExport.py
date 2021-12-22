from ReCal.TransformedRunner import TransformedRunner


class LogitsMemoryExport:
  def __init__(self, loader, loader_str, transformation_specs, n_trans, normalization_params, model, device, transformations=None):
    self.loader = loader
    self.loader_str = loader_str

    self.transformed_runner = TransformedRunner(self.loader, transformation_specs, n_trans, normalization_params, model, device, transformations=transformations)

    self.logits_all_t = {}
    self.y_all = None

  @staticmethod
  def _gen_key(tran_type, tran_arg):
    return tran_type, tran_arg,

  def export(self):
    for idx, (tran_type, tran_arg) in enumerate([('identity', 1.0)] + self.transformed_runner.transformations):
      logits_all, y_all = self.transformed_runner.run_on_tran(self.loader, tran_type, tran_arg)
      self.logits_all_t[LogitsMemoryExport._gen_key(tran_type, tran_arg)] = logits_all

      if tran_type == 'identity':
        self.y_all = y_all

  def export_data(self, xs):
    logits_all_t = {}

    for idx, (tran_type, tran_arg) in enumerate([('identity', 1.0)] + self.transformed_runner.transformations):
      logits_all = self.transformed_runner.run_on_data(xs, tran_type, tran_arg)
      logits_all_t[LogitsMemoryExport._gen_key(tran_type, tran_arg)] = logits_all

    return logits_all_t
