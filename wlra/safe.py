"""Numerically safe functions"""
import numpy as np

def safe_log(x):
  """Numerically safe log"""
  return np.log(x + 1e-8)

def safe_exp(x):
  """Numerically safe exp"""
  x = np.array(x)
  return np.where(x > 100, x, np.exp(x))
