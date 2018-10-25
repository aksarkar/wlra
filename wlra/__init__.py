import torch.cuda

if torch.cuda.is_available():
  from .torch import lra, wlra, plra
else:
  from .wlra import lra, wlra, plra
