# Audio classifier
This is a project developed using Python3 to classify some audio files from Freesound (see "Dataset" below). Libraries used for this project includes PyTorch, Numpy and Librosa and H5py.

## Installation
1. Install [PyTorch](https://pytorch.org/)
2. Install additional packages by running:
```
pip install -r requirements.txt
```

## Dataset
Download the dataset from [Freesound Dataset Kaggle 2018](https://zenodo.org/record/2552860#.XYOJZSgzaHt) and extract the files as follows:
- Labels (2 CSV files): <root>/input/labels
- Train data (~9.5k WAV files): <root>/input/train
- Test data (~1.6k WAV files): <root>/input/test

## Configuration
No arguments required. Run the following command for command-line options
```
python ac.py --help
```

## Run
To run the autoencoder, setup your configuration file and run:
```
python autoencoder.py
```