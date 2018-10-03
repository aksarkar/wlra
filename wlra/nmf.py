import numpy as np
from .safe import *

def frob_loss(x, lf):
  """Return the Frobenius norm ||x - LF||_F

  This implementation supports masked arrays.

  """
  return np.square(x - lf).mean()  

def i_loss(x, lf):
  """Return the I-divergence sum(X log X - X log(LF) - X + LF)

  If we constrain sum(A) = sum(LF) = 1, this is proportional to
  KL(A||LF). Minimizing this loss is then also equivalent to maximizing the
  Poisson likelihood x_ij ~ Pois([LF]_ij).

  For numerical purposes, returns the mean instead of the sum.

  This implementation supports masked arrays.

  """
  return (x * safe_log(x) - x * safe_log(lf) - x + lf).mean()

def nmf(x, rank, frob=True, max_iters=1000, atol=1e-4, verbose=False):
  """Non-negative matrix factorization (Lee and Seung 2001).

  This implementation supports masked arrays.

  """
  n, p = x.shape
  # Random initialization (c.f. https://github.com/scikit-learn/scikit-learn/blob/bac89c2/sklearn/decomposition/nmf.py#L315)
  scale = np.sqrt(x.mean() / rank)
  l = np.random.uniform(0, scale, size=(n, rank))
  f = np.random.uniform(0, scale, size=(rank, p))
  res = l.dot(f)

  if frob:
    loss = frob_loss
  else:
    loss = i_loss
  obj = loss(x, res)
  if verbose:
    print(f'nmf [0]: {obj}')

  for i in range(max_iters):
    if frob:
      f *= l.T.dot(x) / l.T.dot(l).dot(f)
      l *= x.dot(f.T) / l.dot(f).dot(f.T)
    else:
      # F_kj *= L_ik X_ij / [LF]_ij / L_ik
      # L_ik *= F_kj X_ij / [LF]_ij / F_ik
      raise NotImplementedError
    res = l.dot(f)
    update = loss(x, res)
    if verbose:
      print(f'nmf [{i + 1}]: {update}')
    if update > obj and not np.isclose(update, obj, atol=atol):
      raise RuntimeError('objective function increased')
    elif np.isclose(update, obj, atol=atol):
      return res
    else:
      obj = update
  raise RuntimeError('failed to converge')
