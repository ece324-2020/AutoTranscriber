# This file defines the data loaders to use

'''
Provides helper functions and classes for music data loaders.

Usage:

    from train_val_loaders import *

    mean, std = get_statistics("data_path")
    # data_path should be the directory that contains all of the npy files

    train_loader, val_loader, test_loader = get_data_loaders("train_path", "val_path", "test_path", batch_size=batch_size, shuffle=True, normalize=True, mean=mean, std=std)
    # You can load data from train_path and/or val_path and/or test_path.
    # The paths should point to a directory with a 'data' and 'labels' folder inside.
    for data, label in train_loader:
        print(data.shape, label.shape)
'''

import numpy as np
import os
import pretty_midi
from torch.utils.data import Dataset, DataLoader

def get_statistics(data_path):

    '''
    Returns the mean and standard deviation across all of the data in
    train_path.

        Parameters:
            train_path (str): Path to directory containing all of the training
                data as .npy files

        Returns:
            mean (float): The mean of all the values in the training data
            std (float): The standard deviation of all of the values in the
                training data
    '''

    if type(data_path) != str:
        raise TypeError('Input argument must be a string.')
    if os.path.isdir(data_path) == False:
        raise ValueError('Input argument must point to a valid directory.')

    data = []
    
    # Store all training data in one list
    for file in os.listdir(data_path):

        if file.endswith('.npy'):
            data.append(np.load(os.path.join(data_path,file)))

    # Compute the mean and standard deviation
    data_array = np.stack(data)
    mean = np.mean(data_array)
    std = np.std(data_array)

    return mean, std

class _MusicDataset(Dataset):

    '''
    A Dataset object for music that returns a 2D spectrogram and its
    corresponding MIDI file when used in a DataLoader.

    Attributes
    ----------
    dir (str): Path to where all of the data (npy files) is. It is expected to
        end with 'data'.
    list_of_data (list): List of all of the data located at dir
    fs (int/float): Sampling frequency to use for MIDI file
    normalize (boolean): Specify to normalize data
    mean (int/float): Mean to use to normalize data
    std (int/float): Standard deviation to use to normalize data

    Methods
    -------
    __len__():
        Returns the number of npy files in self.dir.
    __getitem__(index):
        Returns a data sample and its label from self.dir at index index.
    '''

    def __init__(self, directory, sample_freq=31, normalize=False, mean=None, std=None):

        '''
        Initializes all of the attributes.

            Parameters:
                directory (str): Path to all of the data (npy files).
        '''
        
        assert type(directory) == str, 'Input argument must be a string.'
        # Store directory
        count = 0 # number of slashes at the end of directory
        for i in range(len(directory)-1,0,-1):
            if directory[i] == '/':
                count += 1
            elif directory[i] == '\\':
                count += 1
            else:
                break
        self.dir = directory[:len(directory)-count]
        assert os.path.isdir(self.dir) == True, 'Positional argument "directory" must be a valid directory.'
        assert os.path.isdir(os.path.join(self.dir,'data')) == True, 'There must be a "data" directory within the specified path.'
        assert os.path.isdir(os.path.join(self.dir,'labels')) == True, 'There must be a "labels" directory within the specified path.'
        assert type(sample_freq) == int or type(sample_freq) == float, 'Keyword argument "sample_freq" must be an integer or float.'
        assert type(normalize) == bool, 'Keyword argument "normalize" must be a boolean.'
        if normalize:
            assert type(mean) == int or type(mean) == float, 'Keyword argument "mean" must be an integer or float.'
            assert type(std) == int or type(std) == float, 'Keyword argument "std" must be an integer or float.'

        self.list_of_data = []
        for i in os.listdir(os.path.join(self.dir,'data')):
            if i[-3:] == 'npy':
                self.list_of_data.append(i)
        self.fs = sample_freq
        self.normalize = normalize
        self.mean = mean
        self.std = std
        
    def __len__(self):

        '''
        Returns the number of npy files in self.dir.
        '''

        return len(self.list_of_data)

    def __getitem__(self, index):

        '''
        Returns a data sample and its label from self.dir at index index.

            Parameters:
                index (int): The data sample to get.

            Returns:
                data_sample (numpy.ndarray): A spectrogram
                data_label (numpy.ndarray): A piano roll
        '''

        assert type(index) == int, 'Input argument must be an integer'
        # Get data sample
        data_sample = np.load(os.path.join(self.dir,'data',self.list_of_data[index]))
        data_sample = data_sample[:,:126] # 126 is for 4 seconds out of 8
        if self.normalize:
            data_sample = (data_sample-self.mean)/self.std
        # Load data label
        midi = pretty_midi.PrettyMIDI(os.path.join(self.dir,'labels',
                                                   self.list_of_data[index][:-3]+'mid'))
        piano_roll = midi.get_piano_roll(fs=self.fs)[21+30:21+30+40,:124] # 124 is for 4 seconds out of 8
        data_label = np.where(piano_roll > 0, 1, 0)

        return data_sample, data_label

def get_data_loaders(train_path, val_path=None, test_path=None, batch_size=1,
                     shuffle=True, normalize=False, mean=None, std=None):

    '''
    Returns data loaders to use for training a network with music datasets.

        Parameters:
            train_path (str): Path to directory containing all of the training
                data as .npy files
            val_path (str): Path to directory containing all of the validation
                data as .npy files
            test_path (str): Path to directory containing all of the testing
                data as .npy files
            batch_size (int): Batch size to use when loading data
            shuffle (bool): If data when loaded should be shuffled or not
            normalize (bool): If data should be normalized
            mean (int/float): Mean to use to normalize data
            std (int/float): Standard deviation to use to normalize data
            
        Returns:
            train_loader (DataLoader): Data loader with all of the training data
            val_loader (DataLoader): Data loader with all of the validation
                data
            test_loader (DataLoader): Data loader with all of the testing data
    '''

    assert os.path.isdir(train_path) == True, 'Training path must be a valid directory.'
    if val_path != None:
        assert os.path.isdir(val_path) == True, 'Validation path must be a valid directory.'
    if test_path != None:
        assert os.path.isdir(test_path) == True, 'Testing path must be a valid directory.'
    assert type(batch_size) == int, 'Batch_size must be an integer.'
    assert batch_size > 0, 'Batch_size must be greater than 0.'
    assert type(shuffle) == bool, 'Shuffle must be a boolean.'
    assert type(normalize) == bool, 'Normalize must be a boolean.'
    if normalize:
        assert type(mean) == int or type(mean) == float, 'Mean must be an integer or float.'
        assert type(std) == int or type(std) == float, 'Std must be an integer or float.'

    train_set = _MusicDataset(train_path, normalize=normalize, mean=mean, std=std)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)
    if val_path != None:
        val_set = _MusicDataset(val_path, normalize=normalize, mean=mean, std=std)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=shuffle)
    if test_path != None:
        test_set = _MusicDataset(test_path, normalize=normalize, mean=mean, std=std)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=shuffle)

    if val_path == None and test_path == None:
        return train_loader
    elif val_path != None and test_path == None:
        return train_loader, val_loader
    elif val_path == None and test_path != None:
        return train_loader, test_loader
    else:
        return train_loader, val_loader, test_loader
