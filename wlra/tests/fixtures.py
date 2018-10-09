import numpy as np
import pytest

@pytest.fixture
def simulate():
  np.random.seed(0)
  l = np.random.normal(size=(100, 3))
  f = np.random.normal(size=(3, 200))
  eta = l.dot(f)
  eta *= 5 / eta.max()
  x = np.random.poisson(lam=np.exp(eta))
  return x, eta

@pytest.fixture
def simulate_lam_low_rank():
  np.random.seed(0)
  l = np.exp(np.random.normal(size=(100, 1)))
  f = np.exp(np.random.normal(size=(1, 200)))
  lam = l.dot(f)
  x = np.random.poisson(lam=lam)
  return x, lam

