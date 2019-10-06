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
import sys
import argparse
import threading
import select, sys

stop_training = False
done = False
def check_key_input():
  global stop_training
  k = ''
  while k != 'q' and done==False:
    k = input()
  # Don't print anything if execution is done
  if k == 'q':
    print('Breaking execution after current epoch', end='', flush=True)
  stop_training = True

def setup_output(path_out, labels_train, num_samples_train, num_samples_test):
  # Load one of the WAV files so we can generate a melspectrogram and get the shape from there
  label = labels_train.iloc[0]['label']
  fname = labels_train.iloc[0]['fname']
  path = os.path.join(dir_train, fname)
  melspec, _ = audio.melspectrogram_from_file(path, label)
  groups = []
  # Labels are integers, ranging from zero to (number of labels-1)
  groups.append(storage.HDF5Dataset('train', 'output_data', (num_samples_train, *melspec.shape), 'f'))
  groups.append(storage.HDF5Dataset('train', 'output_label', (num_samples_train,), 'i'))
  groups.append(storage.HDF5Dataset('test', 'output_data', (num_samples_test, *melspec.shape), 'f'))
  groups.append(storage.HDF5Dataset('test', 'output_label', (num_samples_test,), 'i'))
  storage.create_hdf5(path_out, groups)

def setup_labels(path_labels_train, path_labels_test, only_manually_verified):
  labels_train = pd.read_csv(path_labels_train)
  # The Freesound dataset have a number of samples (~60%) that are automatically labelled.
  # These might be incorrect, so we might get better results from excluding these.
  if only_manually_verified==True:
    labels_train = labels_train[labels_train['manually_verified']==1]
  labels_test = pd.read_csv(path_labels_test)
  label_id_to_label = {}
  label_to_label_id = {}
  unique_labels = list(set(labels_train['label']))
  for idx, label in enumerate(unique_labels):
    label_to_label_id[label] = idx
    label_id_to_label[idx] = label

  return labels_train, labels_test, label_id_to_label, label_to_label_id

def create_melspectrograms(path_out, dir_train, dir_test, num_samples_train, num_samples_test, labels_train, labels_test, label_to_label_id):
  setup_output(path_out, labels_train, num_samples_train, num_samples_test)
  with h5py.File(path_out, 'r+') as f:
    dataset_data_train = f['train']['output_data']
    dataset_label_train = f['train']['output_label']
    dataset_data_test = f['test']['output_data']
    dataset_label_test = f['test']['output_label']
    f.attrs['label_to_label_id'] = ','.join([(str(k)+':'+str(v)) for k, v in label_to_label_id.items()])
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
    std = None
    mean = None
    for g in group_data:
      for idx in range(g['num_samples']):
        label = g['labels'].iloc[idx]['label']
        fname = g['labels'].iloc[idx]['fname']
        path = os.path.join(g['path'], fname)
        melspec, sample_rate = audio.melspectrogram_from_file(path, label)
        # if idx==0:
        #   audio.print_melspectrogram(melspec_normalized, sample_rate, 512, 20, 8000)
        # Store data to the first (index 0) channel
        g['dataset_data'][idx] = melspec
        g['dataset_label'][idx] = label_to_label_id[label]
      # NB Make sure the train dataset is first, so we can perform normalization on that dataset
      if std==None:
        mean = np.mean(g['dataset_data'])
        std = np.std(g['dataset_data'])
      g['dataset_data'] = (g['dataset_data']-mean)/std
      mean2 = np.mean(g['dataset_data'])
      std2 = np.std(g['dataset_data'])
      print(f'Mean: {mean} std: {std} Mean2: {mean2} std2: {std2}')

def train(net, dataset_train_data, dataset_train_label, num_targets, num_epochs, batch_size, device):
  global stop_training
  optimizer = optim.Adam(net.parameters(), lr=1e-3)
  samples_train = dataset_train_data[()]
  # Add an extra dimension (channel dimension) required by the CNN
  samples_train = np.expand_dims(samples_train, axis=1)
  samples_train = torch.from_numpy(samples_train).float()
  #audio.print_melspectrogram(samples[0][0], 44100, 512, 20, 8000)
  #plt.imshow(dataset_train_data[0],aspect='auto',origin='lower')
  targets_train = dataset_train_label[()]
  targets_train = torch.from_numpy(targets_train).long()
  lossfn = nn.CrossEntropyLoss()
  net.train()
  num_batches = (len(samples_train)-1)//batch_size+1
  print('Num batches:',num_batches)
  for idx_epoch in range(num_epochs):
    running_loss = 0
    for idx_batch in range(num_batches):
      start = idx_batch*batch_size
      end = (idx_batch+1)*batch_size
      samples = samples_train[start:end].to(device)
      targets = targets_train[start:end].to(device)
      optimizer.zero_grad()
      output = net(samples)
      loss = lossfn(output, targets)
      loss.backward()
      optimizer.step()
      print('*',end='',flush=True)
      running_loss += loss.item()
    print(f'\nDone with epoch {idx_epoch+1}/{num_epochs}. Loss: {running_loss/num_batches:.5f}')
    if stop_training==True:
      break


def test(net, dataset_test_data, dataset_test_label, label_id_to_label, batch_size, device):
  samples_test = dataset_test_data[()]
  samples_test = np.expand_dims(samples_test, axis=1)
  samples_test = torch.from_numpy(samples_test).float()
  targets_test = dataset_test_label[()]
  targets_test = torch.from_numpy(targets_test).long()
  num_batches = (len(samples_test)-1)//batch_size+1
  net.eval()
  with torch.no_grad():        
    target_total = np.zeros(num_targets)
    target_correct = np.zeros(num_targets)
    for idx_batch in range(num_batches):
      start = idx_batch*batch_size
      end = (idx_batch+1)*batch_size
      samples = samples_test[start:end].to(device)
      targets = targets_test[start:end].to(device)
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

def parse_command_line():
  parser = argparse.ArgumentParser()
  parser.add_argument("--rerun", "-r", action='store_const', const=True, default=False, help="Regenerate all melspectrograms (default behavior: reuse generated melspectrograms)")
  parser.add_argument("--num_epochs", "-e", default=20, type=int, required=False, help="Number of epochs used for training")
  parser.add_argument("--batch_size", "-bs", default=1000, type=int, required=False, help="Batch size used during training")
  parser.add_argument("--num_samples", "-n", default=np.inf, type=int, required=False, help="Number of samples to use for training and testing. NUM_SAMPLES = min(NUM_SAMPLES, actual number of samples)")
  parser.add_argument("--include_all", "-i", action='store_const', const=True, default=False, help="Include automatically labelled samples when training (default: False)")
  args = parser.parse_args()
  return args

if __name__=='__main__':
  args = parse_command_line() 
  root = 'data/'
  dir_train = os.path.join(root, 'input/train')
  dir_test = os.path.join(root, 'input/test')
  dir_out = os.path.join(root, 'output')
  path_labels_train = os.path.join(root, 'input/labels/train_post_competition.csv')
  path_labels_test = os.path.join(root, 'input/labels/test_post_competition_scoring_clips.csv')
  path_out = os.path.join(dir_out, 'out.hdf5')
  before = time.time()

  if not os.path.exists(path_out) or args.rerun:
    print('Start generating melspectrograms')
    utils.ensure_directory_existance([dir_out])
    labels_train, labels_test, label_id_to_label, label_to_label_id = setup_labels(path_labels_train, path_labels_test, not args.include_all)
    num_samples_train = labels_train.shape[0]
    num_samples_test = labels_test.shape[0]
    num_samples_train = int(np.minimum(num_samples_train, args.num_samples))
    num_samples_test = int(np.minimum(num_samples_test, args.num_samples))
    create_melspectrograms(path_out, dir_train, dir_test, num_samples_train, num_samples_test, labels_train, labels_test, label_to_label_id)
  else:
    print('Found existing melspectrograms')
  print('Start training')
  t = threading.Thread(target=check_key_input)
  t.start()
  with h5py.File(path_out,'r') as f:
    dataset_train_data = f['train']['output_data']
    dataset_train_label = f['train']['output_label']
    dataset_test_data = f['test']['output_data']
    dataset_test_label = f['test']['output_label']
    label_to_label_id = f.attrs['label_to_label_id']
    label_to_label_id = label_to_label_id.split(',')
    label_to_label_id = dict(x.split(':') for x in label_to_label_id)
    num_samples_train = len(dataset_train_data)
    print(f'Number of train samples: {num_samples_train}')
    label_id_to_label = inv_map = {int(v): k for k, v in label_to_label_id.items()}
    num_targets = len(label_id_to_label.keys())
    num_epochs = args.num_epochs
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f'Hardware used for processing: {device}')
    net = ConvNet(num_targets, device).to(device)
    train(net, dataset_train_data, dataset_train_label, num_targets, num_epochs, args.batch_size, device)
    test(net, dataset_test_data, dataset_test_label, label_id_to_label, args.batch_size, device)

  after = time.time()
  t_tot = after-before
  t_avg = t_tot/num_epochs
  print(f'Time: {t_tot:.2f}s Time/epoch: {t_avg:.2f}s')
  done = True
  if stop_training == False:
    print("Press Enter to quit")