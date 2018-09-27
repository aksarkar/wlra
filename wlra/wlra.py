"""Weighted low rank approximation

Author: Abhishek Sarkar <aksarkar@alum.mit.edu>

"""

import numpy as np
import scipy.special as sp
import scipy.stats as st
import sklearn.decomposition as skd

def lra(x, rank):
  """Return the unweighted low rank approximation of x

  The solution is given by truncated SVD. This implementation automatically
  chooses a randomized algorithm if x is big enough.

  """
  u, d, vt = skd.PCA(n_components=rank)._fit(x)
  if d.shape[0] > rank:
    # It was faster to perform full SVD, so we need to truncate ourselves
    u = u[:,:rank]
    d = d[:rank]
    vt = vt[:rank]
  return np.einsum('ij,j,jk->ik', u, d, vt)

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
  z = np.zeros(x.shape)
  obj = np.inf
  for i in range(max_iters):
    z1 = lra(w * x + (1 - w) * z, rank)
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
  lam = x.mean(axis=0, keepdims=True)
  if np.ma.is_masked(x):
    # This only removes the mask
    lam = lam.filled(0)
  eta = np.ones(x.shape) * np.log(lam)
  for i in range(max_outer_iters):
    lam = np.exp(eta)
    w = lam / 2
    # Important: WLRA requires weights 0 <= w <= 1
    w /= w.max()
    target = eta - x / lam + 1
    if np.ma.is_masked(x):
      # Mark missing data with weight 0
      w *= (~x.mask).astype(int)
      # Now we can go ahead and fill in the missing values with something
      # computationally convenient, because the WLRA EM update will ignore the
      # value for weight zero.
      target = target.filled(0)
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
