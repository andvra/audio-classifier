import numpy as np
import pandas as pd
import librosa, librosa.display
import os
import matplotlib.pyplot as plt
import h5py
import time
import audio
import storage
import utils
from convnet import ConvNet
import torch
import torch.nn as nn
import torch.optim as optim 

def setup_output(path_out, labels_train, num_samples_train, num_samples_test, num_targets):
  # Load one of the WAV files so we can generate a melspectrogram and get the shape from there
  label = labels_train.iloc[0]['label']
  fname = labels_train.iloc[0]['fname']
  path = os.path.join(dir_train, fname)
  melspec, _ = audio.melspectrogram_from_file(path, label)
  groups = []
  # Targets (labels) are one-hot encoded
  # Our ConvNet need the channels dimension. It's always 1 for our melspecs
  num_channels = 1
  groups.append(storage.HDF5Dataset('train', 'output_data', (num_samples_train, *melspec.shape), 'f'))
  groups.append(storage.HDF5Dataset('train', 'output_label', (num_samples_train, num_targets), 'i'))
  groups.append(storage.HDF5Dataset('test', 'output_data', (num_samples_test, *melspec.shape), 'f'))
  groups.append(storage.HDF5Dataset('test', 'output_label', (num_samples_test,num_targets), 'i'))
  storage.create_hdf5(path_out, groups)

def setup_labels(path_labels_train, path_labels_test):
  labels_train = pd.read_csv(path_labels_train)
  print(labels_train.head())
  labels_test = pd.read_csv(path_labels_test)
  label_to_label_id = {}
  label_id_to_label = {}
  label_id_to_target = {}
  label_to_target = {}
  unique_labels = list(set(labels_train['label']))
  for idx, l in enumerate(unique_labels):
    label_to_label_id[l] = idx
    label_id_to_label[idx] = l
    target = np.zeros(len(unique_labels))
    target[idx] = 1
    label_id_to_target[idx] = target
    label_to_target[l] = target

  return labels_train, labels_test, label_to_label_id, label_id_to_label, label_id_to_target, label_to_target

def create_melspectrograms(path_out, dir_train, dir_test, num_samples_train, num_samples_test, labels_train, labels_test):
  with h5py.File(path_out, 'r+') as f:
    dataset_data_train = f['train']['output_data']
    dataset_label_train = f['train']['output_label']
    dataset_data_test = f['test']['output_data']
    dataset_label_test = f['test']['output_label']
    group_data = []
    group_data.append({
      'path': dir_train,
      'dataset_data': dataset_data_train,
      'dataset_label': dataset_label_train,
      'num_samples': num_samples_train,
      'labels': labels_train})
    group_data.append({
      'path': dir_test,
      'dataset_data': dataset_data_test,
      'dataset_label': dataset_label_test,
      'num_samples': num_samples_test,
      'labels': labels_test})
    for g in group_data:
      for idx in range(g['num_samples']):
        label = g['labels'].iloc[idx]['label']
        fname = g['labels'].iloc[idx]['fname']
        path = os.path.join(g['path'], fname)
        melspec, sample_rate = audio.melspectrogram_from_file(path, label)
        melspec_normalized = utils.normalize(melspec)
        # if idx==0:
        #   audio.print_melspectrogram(melspec_normalized, sample_rate, 512, 20, 8000)
        # Store data to the first (index 0) channel
        g['dataset_data'][idx] = melspec_normalized
        g['dataset_label'][idx] = label_to_target[label]

def train(path_out, label_id_to_label, num_targets, num_epochs):
  device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
  net = ConvNet(num_targets, device).to(device)
  optimizer = optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
  with h5py.File(path_out,'r') as f:
    dataset_train_data = f['train']['output_data']
    dataset_train_label = f['train']['output_label']
    dataset_test_data = f['test']['output_data']
    dataset_test_label = f['test']['output_label']
    for idx in range(5):
      label_one_hot = dataset_train_label[idx]
      label_id, = np.where(dataset_train_label[idx]==1)
      label = label_id_to_label[label_id[0]]
      # print(f'OneHot: {dataset_train_label[idx]} ID: {label_id} Label: {label}')
    samples_train = dataset_train_data[()]
    # Add an extra dimension (channel dimension) required by the CNN
    samples_train = np.expand_dims(samples_train, axis=1)
    samples_train = torch.from_numpy(samples_train).float()
    samples_test = dataset_test_data[()]
    samples_test = np.expand_dims(samples_test, axis=1)
    samples_test = torch.from_numpy(samples_test).float()
    #audio.print_melspectrogram(samples[0][0], 44100, 512, 20, 8000)
    #plt.imshow(dataset_train_data[0],aspect='auto',origin='lower')
    targets_train = dataset_train_label[()]
    targets_train = torch.from_numpy(targets_train).float()
    targets_test = dataset_test_label[()]
    targets_test = torch.from_numpy(targets_test).float()
    targets_test = torch.argmax(targets_test, 1)
    lossfn = nn.MSELoss()
    net.train()
    batch_size = 1000
    num_batches = (len(samples_train)-1)//batch_size+1
    print('Num batches:',num_batches)
    for idx_epoch in range(num_epochs):
      running_loss = 0
      for idx_batch in range(num_batches):
        start = idx_batch*batch_size
        end = (idx_batch+1)*batch_size
        samples = samples_train[start:end]
        targets = targets_train[start:end]
        optimizer.zero_grad()
        output = net(samples)
        loss = lossfn(output, targets)
        loss.backward()
        optimizer.step()
        print('*',end='',flush=True)
        running_loss += loss.item()
      print(f'\nDone with epoch {idx_epoch+1}/{num_epochs}. Loss: {running_loss/num_batches:.5f}')
    num_batches = (len(samples_test)-1)//batch_size+1
    with torch.no_grad():        
      target_total = np.zeros(num_targets)
      target_correct = np.zeros(num_targets)
      for idx_batch in range(num_batches):
        start = idx_batch*batch_size
        end = (idx_batch+1)*batch_size
        samples = samples_test[start:end]
        targets = targets_test[start:end]
        output = net(samples)
        output = torch.argmax(output, 1)
        for idx in range(len(output)):
          target_total[targets[idx]] += 1
          if targets[idx]==output[idx]:
            target_correct[targets[idx]] +=1
      for idx in range(num_targets):
        label = label_id_to_label[idx]
        percent = 100*target_correct[idx]/np.maximum(target_total[idx],1)
        print(f'{label:>25}: {percent:.2f}%')
      tot_all = np.sum(target_total)
      tot_correct = np.sum(target_correct)
      tot = 100*tot_correct/tot_all
      print(f'Total: {tot:.2f}% ({tot_correct:.0f}/{tot_all:.0f})')



if __name__=='__main__':
  root = 'data/'
  dir_train = os.path.join(root, 'input/train')
  dir_test = os.path.join(root, 'input/test')
  dir_out = os.path.join(root, 'output')
  path_labels_train = os.path.join(root, 'input/labels/train_post_competition.csv')
  path_labels_test = os.path.join(root, 'input/labels/test_post_competition_scoring_clips.csv')
  path_out = os.path.join(dir_out, 'out.hdf5')

  utils.ensure_directory_existance([dir_out])
  labels_train, labels_test, label_to_label_id, label_id_to_label, label_id_to_target, label_to_target = setup_labels(path_labels_train, path_labels_test)

  num_samples_train = labels_train.shape[0]
  num_samples_test = labels_test.shape[0]
  num_targets = len(label_to_target.keys())
  # TODO: Just use a subset while developing
  #num_samples_train = 100
  #num_samples_test = 100
  num_epochs = 20

  print(f'Number of train samples: {num_samples_train}')
  
  before = time.time()

  setup_output(path_out, labels_train, num_samples_train, num_samples_test, num_targets)
  create_melspectrograms(path_out, dir_train, dir_test, num_samples_train, num_samples_test, labels_train, labels_test)
  print('Start training')
  train(path_out, label_id_to_label, num_targets, num_epochs)

  after = time.time()
  t_tot = after-before
  t_avg = t_tot/num_samples_train
  t_est = t_avg * labels_train.shape[0]
  print(f'Time: {t_tot:.2f}s Avg: {t_avg:.2f}s Est: {t_est:.2f}s')