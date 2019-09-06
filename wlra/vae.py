import torch

class Encoder(torch.nn.Module):
  """Encoder q(z | x) = N(mu(x), sigma^2(x))"""
  def __init__(self, input_dim, output_dim):
    super().__init__()
    self.net = torch.nn.Sequential(
      torch.nn.Linear(input_dim, 128),
      torch.nn.ReLU(),
      torch.nn.Linear(128, 128),
      torch.nn.ReLU(),
    )
    self.mean = torch.nn.Linear(128, output_dim)
    self.scale = torch.nn.Sequential(torch.nn.Linear(128, output_dim), torch.nn.Softplus())

  def forward(self, x):
    q = self.net(x)
    return self.mean(q), self.scale(q)

class Pois(torch.nn.Module):
  """Decoder p(x | z) ~ Poisson(s_i lambda(z))"""
  def __init__(self, input_dim, output_dim):
    super().__init__()
    self.lam = torch.nn.Sequential(
      torch.nn.Linear(input_dim, 128),
      torch.nn.ReLU(),
      torch.nn.Linear(128, 128),
      torch.nn.ReLU(),
      torch.nn.Linear(128, output_dim),
      torch.nn.Softplus(),
    )

  def forward(self, x):
    return self.lam(x)

def kl_term(mean, scale):
  """KL divergence between N(mean, scale) and N(0, 1)"""
  return .5 * (1 - 2 * torch.log(scale) + (mean * mean + scale * scale))

def pois_llik(x, mean):
  """Log likelihood of x distributed as Poisson"""
  return x * torch.log(mean) - mean - torch.lgamma(x + 1)

class PVAE(torch.nn.Module):
  def __init__(self, input_dim, latent_dim):
    super().__init__()
    self.encoder = Encoder(input_dim, latent_dim)
    self.decoder = Pois(latent_dim, input_dim)

  def loss(self, x, s, stoch_samples):
    mean, scale = self.encoder.forward(x)
    # [batch_size]
    # Important: this is analytic
    kl = torch.sum(kl_term(mean, scale), dim=1)
    # [stoch_samples, batch_size, latent_dim]
    qz = torch.distributions.Normal(mean, scale).rsample(stoch_samples)
    # [stoch_samples, batch_size, input_dim]
    lam = self.decoder.forward(qz)
    error = torch.mean(torch.sum(pois_llik(x, lam), dim=2), dim=0)
    # Important: optim minimizes
    loss = -torch.sum(error - kl)
    return loss

  def fit(self, x, s, max_epochs, verbose=False, stoch_samples=10, **kwargs):
    """Fit the model

    :param x: torch.tensor [n_cells, n_genes]
    :param s: torch.tensor [n_cells, 1]

    """
    if torch.cuda.is_available():
      # Move the model and data to the GPU
      self.cuda()
      x.cuda()
      s.cuda()
    stoch_samples = torch.Size([stoch_samples])
    opt = torch.optim.Adam(self.parameters(), **kwargs)
    for epoch in range(max_epochs):
      opt.zero_grad()
      loss = self.loss(x, s, stoch_samples)
      loss.backward()
      opt.step()
      if verbose and not i % 10:
        print(f'[epoch={epoch} batch={i}] elbo={-loss}')
    return self

  @torch.no_grad()
  def denoise(self, x):
    # Plug E[z | x] into the decoder
    return self.decoder.forward(self.encoder.forward(x)[0]).numpy()
