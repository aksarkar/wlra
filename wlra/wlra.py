"""Weighted low rank approximation

Author: Abhishek Sarkar <aksarkar@alum.mit.edu>

"""

import numpy as np
import scipy.special as sp
import sklearn.decomposition as skm

def wlra(x, w, rank, max_iters=10, verbose=False):
  """Return the weighted low rank approximation of x

  Minimize the weighted Frobenius norm between x and the approximation z,
  constraining z to the specificed rank, using EM (Srebro and Jaakola 2003)

  x - input data (n, p)
  w - input weights (n, p)
  rank - rank of the approximation (non-negative)
  max_iters - maximum number of EM iterations. Raises RuntimeError on convergence failure
  verbose - print objective function updates

  """
  n, p = x.shape
  low_rank = x
  pca = skd.PCA(n_components=rank)
  obj = np.inf
  for i in range(max_iters):
    u, d, vt = pca._fit(w * x + (1 - w) * low_rank)
    low_rank = np.einsum('ij,j,jk->ik', u, d, vt)
    update = (w * np.square(x - low_rank)).mean()
    if verbose:
      print(f'wsvd [{i}] = {update}')
    if update > obj or np.isclose(update, obj):
      return low_rank
    else:
      obj = update
  raise RuntimeError('failed to converge')

def pois_llik(y, eta):
  """Return ln p(y | eta) assuming y ~ Poisson(exp(eta))"""
  return y * eta - np.exp(eta) - sp.gammaln(y + 1)

def pois_lra(x, rank, max_outer_iters=10, max_iters=1000, verbose=False):
  """Return the low rank approximation of x assuming Poisson data

  Assume x_ij ~ Poisson(exp(eta_ij)), eta_ij = U_ik D_kk V'_kj. 

  Maximize the log likelihood by using Taylor approximation to rewrite the
  problem as WLRA.

  """
  n, p = x.shape
  eta0 = x.copy()
  for i in range(max_outer_iters):
    lam = np.exp(eta0)
    w = -lam / 2
    target = eta0 - x / lam + 1
    u, d, vt = wlra(target, w, rank, max_iters=max_iters, verbose=verbose)
    eta = np.einsum('ij,j,jk->ik', u, d, vt)
    update = pois_llik(x, eta).mean()
    if verbose:
      print(f'srebro_pois_svd [{i}]: {update}')
    if update > obj or np.isclose(update, obj):
      return eta
    else:
      eta0 = eta
  raise RuntimeError('failed to converge')
