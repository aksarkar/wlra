import numpy as np
import pytest
import scipy.stats as st
import sklearn.decomposition as skd

from fixtures import simulate
from wlra.nmf import nmf

def test_nmf_shape(simulate):
  x, eta = simulate
  res = nmf(x, 1)
  assert res.shape == x.shape

def test_nmf_rank(simulate):
  x, eta = simulate
  res = nmf(x, 1)
  u, d, vt = np.linalg.svd(res)
  assert (~np.isclose(d, 0)).sum() == 1

def test_nmf_objective(simulate):
  x, eta = simulate
  res = nmf(x, 1)
  obj = np.linalg.norm(x - res)
  res0 = skd.NMF(n_components=1).fit(x)
  assert np.isclose(obj, res0.reconstruction_err_)

def test_nmf_mask(simulate):
  x, eta = simulate
  mask = np.random.uniform(size=x.shape) < 0.25
  y = np.ma.masked_array(x, mask=mask)
  res = nmf(y, 1)
  res0 = nmf(x, 1)
  assert (res != res0).any()

@pytest.mark.xfail
def test_nmf_not_frob(simulate):
  x, eta = simulate
  res = nmf(x, 1, frob=False)

def test_nmf_return_lf(simulate):
  x, eta = simulate
  l, f = nmf(x, 1, return_lf=True)
  assert l.shape == (x.shape[0], 1)
  assert f.shape == (1, x.shape[1])
