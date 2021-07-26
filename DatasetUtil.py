import numpy as np

from torch.utils.data import Subset


# TODO: will change valid_ratio to valid_cnt
def split_train_val(org_train_set, shuffle=False, valid_ratio=0.1, val_idx=None):
  num_train = len(org_train_set)

  if val_idx is None:
    if valid_ratio < 1:
      split = int(np.floor(valid_ratio * num_train))
    else:
      split = valid_ratio

    indices = list(range(num_train))

    if shuffle:
      np.random.shuffle(indices)

    train_idx, val_idx = indices[split:], indices[:split]

    new_train_set = Subset(org_train_set, train_idx)
    val_set = Subset(org_train_set, val_idx)

    assert num_train - split == len(new_train_set)
    assert split == len(val_set)
  else:
    all_indices = set(list(range(len(org_train_set))))
    train_idx = list(all_indices - set(val_idx))

    new_train_set = Subset(org_train_set, train_idx)
    val_set = Subset(org_train_set, val_idx)

    assert num_train == (len(new_train_set) + len(val_set))

  return new_train_set, val_set, train_idx, val_idx

