import numpy as np
import os

def normalize(X):
  # TODO: For a melspectrogram, it seems max(X)-min(X) is always 80! Why?
  #print(f'Min: {np.min(X)} Max: {np.max(X)}')
  X -= np.min(X)
  #print(f'Min: {np.min(X)} Max: {np.max(X)}')
  X /= np.max(X)
  #print(f'Min: {np.min(X)} Max: {np.max(X)}')
  return X

def ensure_directory_existance(dirs):
  for d in dirs:
    if not os.path.exists(d):
      os.makedirs(d)