'''
Load utility function to show qualitative results from the music network.

Usage:

    from qualitative_result import qualitative_result

    # Load model

    net = amtNetMk4()
    net.to(device)
    net.load_state_dict(torch.load("Net.pt"))
    net.eval()

    # View qualitative result

    data_file, label_sample, prediction = qualitative_result(net, "MAESTRO/test/")

'''

import matplotlib.pyplot as plt
import numpy as np
import os
import pretty_midi
import random
import torch

def qualitative_result(net, path, rand=True):

    '''
    Plots the ground truth label and network's output of a data sample.

        Parameters:
            net (PyTorch model): A neural network
            path (str): Path to directory containing 'data' and 'labels'
                directories if random = True, or path to a specific file if
                random = False
            rand (bool): If True, a random data sample will be selected
                from path. Otherwise, the specified file will be used.

        Returns:
            data_file (str): Name of the data sample used
            label_sample (numpy.ndarray): Ground truth label
            prediction (torch.tensor): The output of the net
    '''

    if type(path) != str:
        raise TypeError('Input "path" must be a string')
    if os.path.exists(path) == False:
        raise ValueError('Input "path" must be a valid path')
    if type(rand) != bool:
        raise TypeError('Input "random" must be a boolean')

    # Load sample s[ectrogram and midi
    
    if rand:

        data_list = os.listdir(os.path.join(path,'data'))
        data_file = data_list[random.randint(0,len(data_list)-1)]
        data_sample = np.load(os.path.join(path,'data',data_file))

        midi = pretty_midi.PrettyMIDI(os.path.join(path,'labels',data_file[:-4]+'.mid')) # assume data_file ends with a three letter file type
        label_sample = midi.get_piano_roll(fs=31)[21:109,:124]
        label_sample = np.where(label_sample > 0, 1, 0)
        
    else:

        len_of_path = len(path)
        for i in range(len_of_path):
            if path[-1] == '/' or path[-1] == '\\':
                path = path[:-1]
            else:
                break
        for i in range(len(path)-1,-1,-1):
            if path[i] == '/' or path[i] == '\\':
                break

        data_file = path
        data_sample = np.load(path)
        midi = pretty_midi.PrettyMIDI(path[:i-5]+'/labels'+path[i:-4]+'.mid') # assume data_file ends with a three letter file type
        label_sample = midi.get_piano_roll(fs=31)[21:109,:124]
        label_sample = np.where(label_sample > 0, 1, 0)
        
    # Get net prediction

    input_tensor = (torch.from_numpy(data_sample)).unsqueeze(0).unsqueeze(0)
    device = next(net.parameters()).device
    prediction = net(input_tensor.to(device))
    prediction = torch.where(prediction > 0, 1, 0)

    # Visualize ground truth label and model prediction
    
    plt.figure(figsize=(12,12))
    plt.subplot(211)
    plt.subplot(211).set_title('Ground Truth MIDI')
    plt.subplot(211).set_xlabel('Frame')
    plt.subplot(211).set_ylabel('MIDI Note')
    plt.imshow(label_sample, cmap="Blues", aspect="auto", origin='lower') # auto makes each box rectangular
    plt.colorbar()
    plt.subplot(212)
    plt.subplot(212).set_title('Network Predictions')
    plt.subplot(212).set_xlabel('Frame')
    plt.subplot(212).set_ylabel('MIDI Note')
    plt.imshow(prediction.squeeze(0).detach().numpy(), cmap="Blues", aspect="auto", origin='lower') # auto makes each box rectangular
    plt.colorbar()
    plt.show()

    return data_file, label_sample, prediction
