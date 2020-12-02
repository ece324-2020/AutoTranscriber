'''
Converts an 88-note piano roll to a MIDI file and saves it.

Usage:

    from piano_roll_to_midi import *
    piano_roll_to_midi('save_path', piano_roll)
'''

from reverse_pianoroll import *
import numpy as np

def piano_roll_to_midi(save_path, piano_roll, fs=31):

    '''
    Converts an 88-note piano roll to a MIDI file and saves it.

        Parameters:
            save_path (str): Where the MIDI file will be saved
            piano_roll (np.ndarray): The piano roll to convert to a MIDI file
            fs (int/float): Sampling frequency to use to convert piano_roll to a
                MIDI

        Returns:
            midi (pretty_midi.PrettyMIDI): Piano_roll as a PrettyMIDI object
    '''

    if type(piano_roll) != np.ndarray:
        raise TypeError('Input piano_roll must be a numpy array')
    if piano_roll.shape[0] != 88:
        raise ValueError('First dimention of input piano_roll must be 88')
    if len(piano_roll.shape) != 2:
        raise ValueError('Input piano_roll must be 2D')
    if type(fs) != int and type(fs) != float:
        raise TypeError('Input fs must be an integer or float')
    if fs <= 0:
        raise ValueError('Input fs must be greater than 0')

    # Rescale piano_roll so maximum value is mezzo forte
    max_vel = piano_roll.max()
    min_vel = piano_roll.min()
    new_piano_roll = (piano_roll-min_vel)/(max_vel-min_vel)*80 # 80 is mezzo forte

    # Pad piano_roll to correct dimension
    padded_roll = np.zeros((128, piano_roll.shape[1]))
    padded_roll[21:109, :] = new_piano_roll

    # Convert to MIDI and save    
    midi = piano_roll_to_pretty_midi(padded_roll, fs=fs)
    midi.write(save_path)

    return midi
