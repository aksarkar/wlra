import numpy as np
import pytest
import sklearn.decomposition as skd
import wlra

def test_wlra_shape():
  x = np.zeros((100, 200))
  w = np.ones((100, 200))
  res = wlra.wlra(x, w, rank=1)
  assert res.shape == (100, 200)

def test_wlra_unit_weight():
  np.random.seed(0)
  x = np.random.normal(size=(100, 200))
  res = wlra.wlra(x, w=1, rank=1)
  res0 = wlra.lra(x, rank=1)
  assert np.isclose(res, res0).all()

def test_wlra_rank_2():
  np.random.seed(0)
  x = np.random.normal(size=(100, 200))
  res = wlra.wlra(x, w=1, rank=2)
  res0 = wlra.lra(x, rank=2)
  assert np.isclose(res, res0).all()

def test_pois_lra_shape():
  x = np.ones((100, 200))
  res = wlra.pois_lra(x, 1, verbose=True)
  assert res.shape == (100, 200)

def test_pois_lra_masked_array():
  np.random.seed(0)
  x = np.random.poisson(lam=np.exp(np.random.normal(size=(100, 200))))
  x = np.ma.masked_equal(x, 1)
  res = wlra.pois_lra(x, 1)
  assert res.shape == (100, 200)
  assert not np.ma.is_masked(res)

def test_hsvd():
  x = np.zeros((100, 200))
  res = wlra.hsvd(x, s=1, rank=1)
  assert res.shape == (100, 200)
