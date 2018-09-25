"""Weighted low rank approximation

Author: Abhishek Sarkar <aksarkar@alum.mit.edu>

"""

import numpy as np
import scipy.special as sp
import sklearn.decomposition as skd

def wlra(x, w, rank, max_iters=1000, atol=1e-3, verbose=False):
  """Return the weighted low rank approximation of x

  Minimize the weighted Frobenius norm between x and the approximation z,
  constraining z to the specificed rank, using EM (Srebro and Jaakola 2003)

  x - input data (n, p)
  w - input weights (n, p)
  rank - rank of the approximation (non-negative)
  max_iters - maximum number of EM iterations. Raises RuntimeError on convergence failure
  atol - minimum absolute difference in objective function for convergence
  verbose - print objective function updates

  Returns:

  low_rank - ndarray (n, p)

  """
  n, p = x.shape
  # Important: the procedure is deterministic, so initialization
  # matters.
  #
  # Srebro and Jaakkola suggest the best strategy is to initialize
  # from zero, but go from a full rank down to a rank k approximation in
  # the first iterations
  #
  # Here, we take the simpler strategy of initializing to zero, which could end
  # in a local optimum and underfit the data.
  z = np.zeros(x.shape)
  pca = skd.PCA(n_components=rank)
  obj = np.inf
  for i in range(max_iters):
    u, d, vt = pca._fit(w * x + (1 - w) * z)
    z1 = np.einsum('ij,j,jk->ik', u, d, vt)
    update = (w * np.square(x - z1)).mean()
    if verbose:
      print(f'wsvd [{i}] = {update}')
    if update > obj:
      return z
    elif np.isclose(update, obj, atol=atol):
      return z1
    else:
      z = z1
      obj = update
  raise RuntimeError('failed to converge')

      obj = update
  raise RuntimeError('failed to converge')

def pois_llik(y, eta):
  """Return ln p(y | eta) assuming y ~ Poisson(exp(eta))

  This implementation supports broadcasting eta (i.e., sharing parameters
  across observations).

  y - scalar or ndarray
  eta - scalar or ndarray

  Returns:

  llik - ndarray (y.shape)

  """
  return y * eta - np.exp(eta) - sp.gammaln(y + 1)

def pois_lra(x, rank, max_outer_iters=10, max_iters=1000, atol=1e-3, verbose=False):
  """Return the low rank approximation of x assuming Poisson data

  Assume x_ij ~ Poisson(exp(eta_ij)), eta_ij = L_ik F_kj

  Maximize the log likelihood by using Taylor approximation to rewrite the
  problem as WLRA.

  x - input data (n, p)
  rank - rank of the approximation
  max_outer_iters - maximum number of calls to WLRA
  max_iters - maximum number of EM iterations in WLRA
  verbose - print objective function updates
  
  Returns:

  eta - low rank approximation (n, p)

  """
  n, p = x.shape
  obj = -np.inf
  eta = np.ones(x.shape) * np.log(x.mean(axis=1, keepdims=True))
  for i in range(max_outer_iters):
    lam = np.exp(eta)
    w = lam / 2
    # TODO: WLRA requires weights 0 <= w <= 1
    w /= w.max()
    target = eta - x / lam + 1
    eta1 = wlra(target, w, rank, max_iters=max_iters, atol=atol, verbose=verbose)
    update = pois_llik(x, eta1).mean()
    if verbose:
      print(f'pois_lra [{i}]: {update}')
    if update < obj:
      return eta
    elif np.isclose(update, obj, atol=atol):
      return eta1
    else:
      eta = eta1
      obj = update
  raise RuntimeError('failed to converge')
