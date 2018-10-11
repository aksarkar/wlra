"""Weighted low rank approximation on the GPU

"""
import torch
import torch.cuda

from .safe import *
from .wlra import pois_llik

@torch.no_grad()
def get_proj(x, rank, num_iters=4):
  if not torch.cuda.is_available():
    raise RuntimeError('cuda is not available')
  with torch.cuda.device(0):
    x = torch.tensor(x, dtype=torch.float)
    n, p = x.shape
    proj = torch.randn(p, rank)
    for _ in range(num_iters):
      proj = torch.qr(torch.mm(x, proj))[0]
      proj = torch.qr(torch.mm(x.transpose(0, 1), proj))[0]
    return torch.qr(torch.mm(x, proj))[0]

@torch.no_grad()
def lra(x, rank, num_oversamples=10):
  if not torch.cuda.is_available():
    raise RuntimeError('cuda is not available')
  with torch.cuda.device(0):
    x = torch.tensor(x, dtype=torch.float)
    n, p = x.shape
    transpose = n < p
    if transpose:
      x = x.transpose(0, 1)
    proj = get_proj(x, rank + num_oversamples)
    target = torch.mm(proj.transpose(0, 1), x)
    u, d, v = torch.svd(target)
    u = torch.mm(proj, u)
    res = torch.mm(torch.mm(u[:,:rank], torch.diag(d[:rank])),
                   v[:,:rank].transpose(0, 1))
    if transpose:
      res = res.transpose(0, 1)
    return res

@torch.no_grad()
def wlra(x, w, rank, max_iters=10000, atol=1e-3, verbose=False):
  if not torch.cuda.is_available():
    raise RuntimeError('cuda is not available')
  with torch.cuda.device(0):
    x = torch.tensor(x, dtype=torch.float)
    w = torch.tensor(w, dtype=torch.float)
    w /= w.max()
    z = torch.zeros(x.shape)
    obj = torch.sum(w * torch.pow(x, 2))
    if verbose:
      print(f'torch_wlra [0] = {obj}')
    for i in range(max_iters):
      z1 = lra(w * x + (1 - w) * z, rank=rank)
      update = torch.sum(w * torch.pow(x - z1, 2))
      if verbose:
        print(f'torch_wlra [{i + 1}] = {update}')
      if update > obj:
        raise RuntimeError('objective increased')
      elif abs(update - obj) < atol:
        return z1
      else:
        z = z1
        obj = update
    raise RuntimeError('objective increased')

@torch.no_grad()
def plra(x, rank, **kwargs):
  """Return the low rank approximation of x assuming Poisson data

  Assume x_ij ~ Poisson(exp(eta_ij)), eta_ij = L_ik F_kj

  Maximize the log likelihood by using Taylor approximation to rewrite the
  problem as WLRA.

  :param x: input data (n, p)
  :param rank: rank of the approximation
  :param kwargs: keyword arguments to wlra

  :returns eta: low rank approximation (n, p)

  """
  if not torch.cuda.is_available():
    raise RuntimeError('cuda is not available')
  eta = np.where(x > 0, safe_log(x), -np.log(2))
  obj = pois_llik(x, eta).sum()
  lam = safe_exp(eta)
  w = lam
  target = eta + x / lam - 1
  if np.ma.is_masked(x):
    # Mark missing data with weight 0
    w *= (~x.mask).astype(float)
    # Now we can go ahead and fill in the missing values with something
    # computationally convenient, because the WLRA EM update will ignore the
    # value for weight zero.
    target = target.filled(0)
  return wlra(target, w, rank)
