import librosa
import matplotlib.pyplot as plt
import numpy as np

def melspectrogram_from_file(path, label):
  global max_len,min_len
  num_seconds = 2
  # Load sample with preserved sample rate (sr=None)
  clip, sample_rate = librosa.load(path, sr=None)
  num_samples_orig = len(clip)
  num_samples_desired = sample_rate*num_seconds
  if num_samples_orig>num_samples_desired:
    # Extract a random part of the clip
    offset = np.random.randint(num_samples_orig-num_samples_desired)
    clip = clip[offset:offset+num_samples_desired]
  elif num_samples_orig<num_samples_desired:
    # Pad the clips with zeros
    offset = np.random.randint(num_samples_desired-num_samples_orig)
    clip = np.pad(clip, (offset, num_samples_desired-offset-num_samples_orig))
  mfcc = librosa.feature.mfcc(clip, sr=sample_rate, n_mfcc=40)
  return mfcc, sample_rate
  
def print_melspectrogram(logS, sample_rate, hop_length, fmin, fmax):
  print(np.min(logS), np.max(logS))
  plt.axis('off')
  plt.axes([0.,0.,1.,1.], frameon=False, xticks=[],yticks=[])
  librosa.display.specshow(logS, sr=sample_rate, hop_length=hop_length, x_axis='time', y_axis='mel', fmin=fmin, fmax=fmax)
  plt.show()
  #plt.savefig(f'asd.png',bbox_inches=None,pad_inches=0)