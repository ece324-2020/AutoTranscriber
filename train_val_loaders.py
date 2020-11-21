# This file defines the data loaders to use

import numpy as np
import os

def get_statistics(train_path):

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

    if type(train_path) != str:
        raise TypeError('Input argument must be a string.')
    if os.path.isdir(train_path) == False:
        raise ValueError('Input argument must point to a valid directory.')

    training_data = []
    
    # Store all training data in one list
    for file in os.listdir(train_path):

        if file.endswith('.npy'):
            training_data.append(np.load(file))

    # Compute the mean and standard deviation
    training_array = np.stack(training_data)
    mean = np.mean(training_array)
    std = np.std(training_array)

    return mean, std

