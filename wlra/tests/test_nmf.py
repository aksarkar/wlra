import numpy as np
import pytest
import sklearn.decomposition as skd

from wlra.nmf import nmf

@pytest.fixture
def simulate():
  np.random.seed(0)
  l = np.random.normal(size=(100, 1))
  f = np.random.normal(size=(1, 200))
  x = np.exp(l.dot(f) + np.random.normal(size=(100, 200)))
  return x

def test_nmf_shape(simulate):
  x = simulate
  res = nmf(x, 1)
  assert res.shape == x.shape

def test_nmf_rank(simulate):
  x = simulate
  res = nmf(x, 1)
  u, d, vt = np.linalg.svd(res)
  assert (~np.isclose(d, 0)).sum() == 1

def test_nmf_objective(simulate):
  x = simulate
  res = nmf(x, 1)
  obj = np.linalg.norm(x - res)
  res0 = skd.NMF(n_components=1).fit(x)
  assert np.isclose(obj, res0.reconstruction_err_)

def test_nmf_mask(simulate):
  x = simulate
  mask = np.random.uniform(size=x.shape) < 0.25
  y = np.ma.masked_array(x, mask=mask)
  res = nmf(y, 1)
  res0 = nmf(x, 1)
  assert (res != res0).any()

@pytest.mark.xfail
def test_nmf_not_frob(simulate):
  x = simulate
  res = nmf(x, 1, frob=False)
