import numpy as np
import os
import pickle
import pytest
import scipy.stats as st
import wlra

from fixtures import *

# This is needed to get functions not publicly exported
from wlra.wlra import lra
from wlra.nmf import nmf

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

def test_wlra_missing(simulate):
  x, eta = simulate
  w = (np.random.uniform(size=x.shape) < 0.1).astype(float)
  wlra.wlra(x, w, rank=3)

def test_plra_shape():
  x = np.ones((100, 200))
  res = wlra.plra(x, 1)
  assert res.shape == (100, 200)

def test_plra_assume_rank_1():
  x = np.random.poisson(lam=np.exp(np.random.normal(size=(100, 200))))
  res = wlra.plra(x, 1)

def test_plra_oracle(simulate):
  x, eta = simulate
  l1 = st.poisson(mu=np.exp(wlra.plra(x, rank=3, max_outer_iters=100, check_converged=True))).logpmf(x).sum()
  l0 = st.poisson(mu=np.exp(eta)).logpmf(x).sum()
  assert l1 > l0
  
def test_plra1_oracle(simulate):
  x, eta = simulate
  l1 = st.poisson(mu=np.exp(wlra.plra(x, rank=3, max_outer_iters=1))).logpmf(x).sum()
  l0 = st.poisson(mu=np.exp(eta)).logpmf(x).sum()
  assert l1 > l0
  
def test_plra_mask(simulate):
  x, eta = simulate
  mask = np.random.uniform(size=x.shape) < 0.25
  x = np.ma.masked_array(x, mask=mask)
  res = wlra.plra(x, 3)

@pytest.mark.skip('dummy test')
def test_plra1_10x():
  import scmodes
  x = scmodes.dataset.read_10x(f'/project2/mstephens/aksarkar/projects/singlecell-ideas/data/10xgenomics/b_cells/filtered_matrices_mex/hg19/', return_df=True)
  res = wlra.plra(x.values, rank=10, verbose=True)
