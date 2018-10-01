import numpy as np

def nmf(x, rank, max_iters=1000, atol=1e-4, verbose=False):
  """Non-negative matrix factorization (Lee and Seung 2001)

  This implementation supports masked arrays.

  """
  n, p = x.shape
  # Random initialization (c.f. https://github.com/scikit-learn/scikit-learn/blob/bac89c2/sklearn/decomposition/nmf.py#L315)
  scale = np.sqrt(x.mean() / rank)
  l = np.abs(np.random.uniform(0, scale, size=(n, rank)))
  f = np.abs(np.random.uniform(0, scale, size=(rank, p)))

  obj = np.inf
  for i in range(max_iters):
    f *= l.T.dot(x) / l.T.dot(l).dot(f)
    l *= x.dot(f.T) / l.dot(f).dot(f.T)
    res = l.dot(f)
    # np.linalg.norm doesn't support masked arrays, so compute Frobenius norm
    # ourselves
    update = np.square(x - res).mean()
    if verbose:
      print(f'nmf [{i}]: {update}')
    if update > obj:
      raise ValueError('objective function increased')
    elif np.isclose(update, obj, atol=atol):
      return res
    else:
      obj = update
  raise ValueError('failed to converge')

