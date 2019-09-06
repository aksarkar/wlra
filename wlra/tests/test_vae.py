import numpy as np
import torch
import pytest
import scipy.stats as st
import wlra.vae

from fixtures import simulate

@pytest.fixture
def dims():
  # Data (n, p); latent representation (n, d)
  n = 50
  p = 1000
  d = 20
  stoch_samples = 10
  return n, p, d, stoch_samples

def test_encoder(dims):
  n, p, d, stoch_samples = dims
  enc = wlra.vae.Encoder(p, d)
  x = torch.tensor(np.random.normal(size=(n, p)), dtype=torch.float)
  mean, scale = enc.forward(x)
  assert mean.shape == (n, d)
  assert scale.shape == (n, d)

def test_decoder(dims):
  n, p, d, stoch_samples = dims
  dec = wlra.vae.Pois(d, p)
  x = torch.tensor(np.random.normal(size=(n, d)), dtype=torch.float)
  lam = dec.forward(x)
  assert lam.shape == (n, p)

def _fit_pvae(x, max_epochs=100):
  n, p = x.shape
  s = x.sum(axis=1, keepdims=True)
  assert s.shape == (n, 1)
  latent_dim = 10
  x = torch.tensor(x, dtype=torch.float)
  s = torch.tensor(s, dtype=torch.float)
  model = wlra.vae.PVAE(p, latent_dim).fit(x, s, lr=1e-2, verbose=False, max_epochs=max_epochs)
  return model, x
  
def test_pvae(simulate):
  x, eta = simulate
  _fit_pvae(x, max_epochs=1)

def test_pvae_denoise(simulate):
  x, eta = simulate
  model, xt = _fit_pvae(x, max_epochs=1)
  lam = model.denoise(xt)
  assert lam.shape == x.shape

def test_pvae_oracle(simulate):
  x, eta = simulate
  l0 = st.poisson(mu=np.exp(eta)).logpmf(x).mean()
  model, xt = _fit_pvae(x, max_epochs=1000)
  lam = model.denoise(xt)
  l1 = st.poisson(mu=lam).logpmf(x).mean()
  assert l1 > l0
