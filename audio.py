import librosa
import matplotlib.pyplot as plt
import numpy as np

def melspectrogram_from_file(path, label):
  # Load sample with preserved sample rate (sr=None)
  clip, sample_rate = librosa.load(path, sr=None)
  # Maximum three seconds (for performance)
  clip = clip[:sample_rate*3]
  orig_len = len(clip)
  # Pad with zeros (= silent) for the rest of the short sample
  if len(clip) < sample_rate*3:
    padded = np.zeros((sample_rate*3,))
    padded[:len(clip)] = clip
    clip = padded
  # print(f'Num samples: {len(clip):7d} ({orig_len:7d}) SR: {sample_rate} Length (s): {len(clip)/sample_rate:6.2f} Label: {label} ({hash(label)})')
  logS = get_melspectrogram(clip, sample_rate)
  return logS, sample_rate

def get_melspectrogram(clip, sample_rate, n_fft=1024, hop_length=512):
  global tot_min, tot_max, all_mean, all_var
  #fmax = sample_rate/2
  # Most of the important information seem to be below 8kHz
  fmax = 8000
  # We basically don't hear anything below 20Hz
  fmin = 20
  n_mels = 64
  S = librosa.feature.melspectrogram(clip, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmin=fmin, fmax=fmax)
  magnitude, phase = librosa.magphase(S)
  logS = librosa.power_to_db(magnitude)
  return logS
  
def print_melspectrogram(logS, sample_rate, hop_length, fmin, fmax):
  print(np.min(logS), np.max(logS))
  plt.axis('off')
  plt.axes([0.,0.,1.,1.], frameon=False, xticks=[],yticks=[])
  librosa.display.specshow(logS, sr=sample_rate, hop_length=hop_length, x_axis='time', y_axis='mel', fmin=fmin, fmax=fmax)
  plt.show()
  #plt.savefig(f'asd.png',bbox_inches=None,pad_inches=0)