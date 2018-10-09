import numpy as np
import scipy.stats as st
import scipy.special as sp
import pytest

from fixtures import *
from test_nmf import simulate
from wlra.grad import PoissonFA

def test_PoissonFA(simulate):
  x, eta = simulate
  n, p = x.shape
  m = PoissonFA(n_samples=n, n_features=p, n_components=1).fit(x)

def test_PoissonFA_log_link(simulate):
  x, eta = simulate
  n, p = x.shape
  m = PoissonFA(n_samples=n, n_features=p, n_components=1, log_link=False).fit(x)

def pois_llik(x, lam):
  # scipy.stats.poisson ran into numerical problems, so compute it ourselves
  const = sp.gammaln(x + 1).sum()
  return x * np.log(lam) - lam + const

def test_PoissonFA_res(simulate):
  x, eta = simulate
  n, p = x.shape

  m0 = PoissonFA(n_samples=n, n_features=p, n_components=1, log_link=False).fit(x, max_epochs=10000)
  lam0 = m0.L.dot(m0.F)
  l0 = pois_llik(x, lam0)

  m1 = PoissonFA(n_samples=n, n_features=p, n_components=1, log_link=True).fit(x, max_epochs=10000)
  lam1 = np.exp(m1.L.dot(m1.F))
  l1 = pois_llik(x, lam1)

  assert np.isfinite(l0).all()
  assert np.isfinite(l1).all()
  assert np.isfinite(l0.sum())
  assert np.isfinite(l1.sum())
  assert l1.sum() > l0.sum()

def test_PoissonFA_res_true_rank(simulate):
  x, eta = simulate
  n, p = x.shape
  rank = (~np.isclose(np.linalg.svd(np.exp(eta), compute_uv=False, full_matrices=False), 0)).sum()

  m0 = PoissonFA(n_samples=n, n_features=p, n_components=rank, log_link=False).fit(x)
  lam0 = m0.L.dot(m0.F)
  l0 = pois_llik(x, lam0)

  m1 = PoissonFA(n_samples=n, n_features=p, n_components=1, log_link=True).fit(x)
  lam1 = np.exp(m1.L.dot(m1.F))
  l1 = pois_llik(x, lam1)

  assert np.isfinite(l0).all()
  assert np.isfinite(l1).all()
  assert np.isfinite(l0.sum())
  assert np.isfinite(l1.sum())
  assert l1.sum() > l0.sum()

def test_PoissonFA_simulate_lam(simulate_lam_low_rank):
  x, lam = simulate_lam_low_rank
  n, p = x.shape
  rank = (~np.isclose(np.linalg.svd(np.log(lam), compute_uv=False, full_matrices=False), 0)).sum()
  assert rank < min(n, p)
  
  m0 = PoissonFA(n_samples=n, n_features=p, n_components=1, log_link=False).fit(x)
  lam0 = m0.L.dot(m0.F)
  l0 = pois_llik(x, lam0)

  m1 = PoissonFA(n_samples=n, n_features=p, n_components=rank, log_link=True).fit(x)
  lam1 = np.exp(m1.L.dot(m1.F))
  l1 = pois_llik(x, lam1)

  assert np.isfinite(l0).all()
  assert np.isfinite(l1).all()
  assert np.isfinite(l0.sum())
  assert np.isfinite(l1.sum())
  assert l0.sum() > l1.sum()
