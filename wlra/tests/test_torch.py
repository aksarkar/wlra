import numpy as np
import torch
import wlra

from fixtures import *
from wlra.torch import get_proj, lra as torch_lra, wlra as torch_wlra, plra as torch_plra

def test_torch_get_proj(simulate):
  x, eta = simulate
  res = get_proj(x, rank=10)
  assert res.shape == (x.shape[0], 10)

def test_torch_lra(simulate):
  x, eta = simulate
  res = torch_lra(x, 3)
  res0 = wlra.lra(x, 3)
  assert res.shape == x.shape
  assert np.linalg.norm(x - res) <= np.linalg.norm(x - res0)

def test_torch_wlra_unit_weight(simulate):
  x, eta = simulate
  res = torch_wlra(x, w=1, rank=3)
  res0 = torch_lra(x, rank=3)
  assert np.isclose(np.linalg.norm(x - res), np.linalg.norm(x - res0))

def test_torch_wlra_cpu(simulate):
  x, eta = simulate
  res0 = wlra.wlra(x, w=1, rank=3)
  res = torch_wlra(x, w=1, rank=3)
  assert np.linalg.norm(x - res) <= np.linalg.norm(x - res0)

def test_torch_wlra_missing(simulate):
  x, eta = simulate
  w = (np.random.uniform(size=x.shape) < 0.1).astype(float)
  torch_wlra(x, w, rank=3, atol=1e-2)
  
def test_torch_plra(simulate):
  x, eta = simulate
  torch_plra(x, rank=3)

def test_torch_plra_missing(simulate):
  x, eta = simulate
  mask = (np.random.uniform(size=x.shape) < 0.1).astype(float)
  x = np.ma.masked_array(x, mask)
  res = torch_plra(x, rank=3)
