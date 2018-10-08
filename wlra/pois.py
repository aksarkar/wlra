import numpy as np
import scipy.special as sp
import torch

class PoissonFA(torch.nn.Module):
  """Poisson factor analysis via first order optimization"""
  def __init__(self, n_samples, n_features, n_components, log_link=True):
    """Initialize the loadings and factors. 

    The shapes need to be specified here to build the computation graph.

    :param n_samples: number of input samples
    :param n_features: number of input features
    :param n_components: rank of the factorization
    :param log_link: parameterize x_ij ~ Pois(exp(L_ik F_kj))
    """
    super().__init__()
    self.log_link = log_link
    self.l = torch.randn([n_samples, n_components], requires_grad=True)
    self.f = torch.randn([n_components, n_features], requires_grad=True)

  def forward(self, x):
    """Return the log likelihood of x assuming x_ij ~ Pois(exp(l_ik f_kj))

    Drop the constant (x_ij)! to simplify.

    """
    if self.log_link:
      lam = torch.exp(torch.matmul(self.l, self.f))
    else:
      lam = torch.matmul(torch.exp(self.l), torch.exp(self.f))
    return -torch.sum(x * torch.log(lam) - lam)

  def fit(self, x, max_epochs=1000, atol=1, verbose=False, **kwargs):
    """Fit the model and return self.

    :param x: data (n_samples, n_features)
    :param max_epochs: maximum number of iterations of gradient descent
    :param verbose: print objective function updates
    :param atol: absolute tolerance for convergence
    :param **kwargs*: keyword arguments to torch.optim.Adam

    :returns: self

    """
    x = torch.tensor(x, dtype=torch.float)
    opt = torch.optim.Adam([self.l, self.f], **kwargs)
    self.obj = np.inf
    for i in range(max_epochs):
      opt.zero_grad()
      loss = self.forward(x)
      loss_np = loss.detach().numpy()
      if verbose and not i % 100:
        print(f'Epoch {i} = {loss_np}')
      if np.isclose(loss_np, self.obj, atol=atol):
        return self
      else:
        self.obj = loss.detach().numpy()
        loss.backward()
        opt.step()
    return self

  @property 
  def L(self):
    res = self.l.detach().numpy()
    if self.log_link:
      return res
    else:
      return np.exp(res)

  @property
  def F(self):
    res = self.f.detach().numpy()
    if self.log_link:
      return res
    else:
      return np.exp(res)

