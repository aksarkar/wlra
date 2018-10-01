import numpy as np
import pytest
import sklearn.decomposition as skd
import wlra

# This is needed to get functions not publicly exported
from wlra.wlra import lra

def test_lra_shape():
  x = np.zeros((100, 200))
  res = lra(x, rank=1)
  assert res.shape == (100, 200)

def test_lra_value():
  np.random.seed(0)
  x = np.random.normal(size=(100, 200))
  res = lra(x, rank=1)
  u, d, vt = np.linalg.svd(x, full_matrices=False)
  res0 = u[:,:1].dot(vt[:1]) * d[0]
  # Important: numpy/scipy give differences which can differ considerably for
  # individual elements. Instead, check that the objective values are close
  assert np.isclose(np.linalg.norm(x - res), np.linalg.norm(x - res0), atol=0.1)

def test_wlra_shape():
  x = np.zeros((100, 200))
  w = np.ones((100, 200))
  res = wlra.wlra(x, w, rank=1)
  assert res.shape == (100, 200)

def test_wlra_unit_weight():
  np.random.seed(0)
  x = np.random.normal(size=(100, 200))
  res = wlra.wlra(x, w=1, rank=1)
  res0 = lra(x, rank=1)
  assert np.isclose(res, res0).all()

def test_wlra_rank_2():
  np.random.seed(0)
  x = np.random.normal(size=(100, 200))
  res = wlra.wlra(x, w=1, rank=2)
  res0 = lra(x, rank=2)
  assert np.isclose(res, res0).all()

def test_pois_lra_shape():
  x = np.ones((100, 200))
  res = wlra.pois_lra(x, 1)
  assert res.shape == (100, 200)

def test_pois_lra_assume_rank_1():
  x = np.random.poisson(lam=np.exp(np.random.normal(size=(100, 200))))
  res = wlra.pois_lra(x, 1)

def test_pois_lra_masked_array():
  np.random.seed(0)
  x = np.random.poisson(lam=np.exp(np.random.normal(size=(100, 200))))
  x = np.ma.masked_equal(x, 1)
  res = wlra.pois_lra(x, 1)
  assert res.shape == (100, 200)
  assert not np.ma.is_masked(res)

def test_pois_lra_mask():
  np.random.seed(0)
  l = np.random.normal(size=(200, 3))
  f = np.random.normal(size=(3, 300))
  eta = l.dot(f)
  x = np.random.poisson(lam=np.exp(eta))
  mask = np.random.uniform(size=x.shape) < 0.25
  x = np.ma.masked_array(x, mask=mask)
  res = wlra.pois_lra(x, 3)
