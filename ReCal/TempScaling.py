import numpy as np


def calibrate(conf, true_idx, tol=1e-6, max_iter=30, num_guess=100):
  conf = conf.transpose()
  conf = np.maximum(conf, 1e-16)
  x = np.log(conf)

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

