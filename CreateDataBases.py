import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy import fft, signal
import scipy
from scipy.io.wavfile import read
import numpy as np

import glob
from typing import List, Dict, Tuple
from tqdm import tqdm
import pickle

from scipy.io import wavfile
import librosa
import numpy as np
import pyaudio
import os
import functions










for i in range(1,3):
    songs = glob.glob(f'Mode{i}/*wav')

    song_name_index = {}
    database: Dict[int, List[Tuple[int, int]]] = {}

# Go through each song, using where they are alphabetically as an id
    for index, filename in enumerate(tqdm(sorted(songs))):
        song_name_index[index] = filename
    # Read the song, create a constellation and hashes
        Fs, audio_input = read(filename)
        constellation = functions.create_constellation(np.array(audio_input).flatten(), Fs,i)
        hashes = functions.create_hashes(constellation, index)

    # For each hash, append it to the list for this hash
        for hash, time_index_pair in hashes.items():
            if hash not in database:
                database[hash] = []
            database[hash].append(time_index_pair)
# Dump the database and list of songs as pickles
    with open(f"database{i}.pickle", 'wb') as db:
        pickle.dump(database, db, pickle.HIGHEST_PROTOCOL)
    with open(f"song_index{i}.pickle", 'wb') as songs:
        pickle.dump(song_name_index, songs, pickle.HIGHEST_PROTOCOL)











#MFCC Data base

def save_mfccs_to_pickle(dataset, file_names, pickle_filename):
    mfcc_dict = {'dataset': dataset, 'file_names': file_names}
    with open(pickle_filename, 'wb') as f:
        pickle.dump(mfcc_dict, f)
    print(f"MFCCs saved to {pickle_filename}")

# Load and preprocess the dataset
for i in range(1,3):
    dataset_directory = f"Mode{i}"
    dataset, file_names = functions.preprocess_database(dataset_directory)

# Save MFCCs and file names into a pickle file
    pickle_filename = f'mfcc_database{i}.pickle'
    save_mfccs_to_pickle(dataset, file_names, pickle_filename)
