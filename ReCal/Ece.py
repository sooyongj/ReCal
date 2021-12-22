import numpy as np


class ECE:
  def __init__(self, n_bin):
    self.n_bin = n_bin
    bin_boundaries = np.linspace(0, 1, n_bin+1)
    self.bin_lowers = bin_boundaries[:-1]
    self.bin_uppers = bin_boundaries[1:]

    self.ece_mat = None
  
  def compute_acc_conf(self, y_pred, y_true, conf):
    acc = np.equal(y_true, y_pred)

    acc_conf = np.zeros((self.n_bin, 3))
    for i, (bin_lower, bin_upper) in enumerate(zip(self.bin_lowers, self.bin_uppers)):
        in_bin = (conf > bin_lower.item()) & (conf <= bin_upper.item())
        acc_conf[i, 0] = in_bin.astype(float).sum()
        if acc_conf[i, 0] > 0:
            acc_conf[i, 1] = acc[in_bin].astype(float).sum()
            acc_conf[i, 2] = conf[in_bin].astype(float).sum()

    return acc_conf

  def get_ECE_mat(self):
    res_mat = np.copy(self.ece_mat)
    ind = res_mat[:, 0] > 0
    res_mat[ind, 1] = np.divide(res_mat[ind, 1], res_mat[ind, 0])
    res_mat[ind, 2] = np.divide(res_mat[ind, 2], res_mat[ind, 0])
    res_mat[:, 0] = np.divide(res_mat[:, 0], np.sum(res_mat[:, 0]))
    return res_mat

  def compute_ECE(self):
    res_mat = np.copy(self.ece_mat)
    ind = res_mat[:, 0] > 0
    res_mat[ind, 1] = np.divide(res_mat[ind, 1], res_mat[ind, 0])
    res_mat[ind, 2] = np.divide(res_mat[ind, 2], res_mat[ind, 0])
    res_mat[:, 0] = np.divide(res_mat[:, 0], np.sum(res_mat[:, 0]))
    res = np.sum(np.multiply(res_mat[:,0], np.absolute(res_mat[:,1]-res_mat[:,2])))
    return res

  def compute_MCE(self):
    res_mat = np.copy(self.ece_mat)
    ind = res_mat[:, 0] > 0
    res_mat[ind, 1] = np.divide(res_mat[ind, 1], res_mat[ind, 0])
    res_mat[ind, 2] = np.divide(res_mat[ind, 2], res_mat[ind, 0])
    res_mat[:, 0] = np.divide(res_mat[:, 0], np.sum(res_mat[:, 0]))
    res = np.max(np.absolute(res_mat[:, 1] - res_mat[:, 2]))
    return res

  def compute_VCE(self):
    ece = self.compute_ECE()
    res_mat = np.copy(self.ece_mat)
    ind = res_mat[:, 0] > 0
    res_mat[ind, 1] = np.divide(res_mat[ind, 1], res_mat[ind, 0])
    res_mat[ind, 2] = np.divide(res_mat[ind, 2], res_mat[ind, 0])
    res_mat[:, 0] = np.divide(res_mat[:, 0], np.sum(res_mat[:, 0]))
    res = np.sum(np.multiply(res_mat[:,0], np.square(np.absolute(res_mat[:,1]-res_mat[:,2]) - ece)))
    return res

  def add_data(self, y_pred, y_true, conf):
    temp_mat = self.compute_acc_conf(y_pred, y_true, conf)
    if self.ece_mat is None:
      self.ece_mat = temp_mat
    else:
      self.ece_mat = self.ece_mat + temp_mat


if __name__ == '__main__':
  e = ECE(15)

  y_pred = np.array([0,1,2,3,4,5,1,2])
  y_true = np.array([0,1,2,2,2,3,1,2])
  conf = np.array([0.4,0.2,0.3,0.5,0.3,0.7,0.8,0.3])

  e.add_data(y_pred,y_true, conf)
  print(e.ece_mat)
  c = e.compute_ECE()
  print(c)
