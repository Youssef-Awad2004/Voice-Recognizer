# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 22:14:07 2023

@author: yi
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 23:03:40 2023

@author: yi
"""
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
# Fs, song = read("C:/Users/yi/Desktop/Data File/GrantMeAccess/Person1/GrantMeAccessSample1.wav")
# N = len(song)
# fft = scipy.fft.fft(song)
# transform_y = 2.0 / N * np.abs(fft[0:N//2])  
# transform_x = scipy.fft.fftfreq(N, 1 / Fs)[:N//2]
# plt.plot(transform_x, transform_y)
# plt.xlim(0, 2000);


def record_audio(duration):
    CHUNK = 1024  # Number of frames per buffer
    FORMAT = pyaudio.paFloat32  # Audio format (32-bit floating-point)
    CHANNELS = 1 # Number of audio channels (1 for mono, 2 for stereo)
    RATE = 44100  # Sample rate
    audio = pyaudio.PyAudio()

    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

    print("Recording...")
    frames = []
    for i in range(0, int(RATE / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(np.frombuffer(data, dtype=np.float32))

    print("Finished recording")
    stream.stop_stream()
    stream.close()
    audio.terminate()

    return np.hstack(frames), RATE





def create_mfcc(audio_file):
    y, sr = librosa.load(audio_file)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return mfcc

def compare_mfcc(mfcc1, mfcc2):
    # Use Euclidean distance for comparison
    distance = np.linalg.norm(mfcc1 - mfcc2)

    # Calculate similarity as the inverse of the distance
    similarity = 1 / (1 + distance)

    return similarity

def compare_mfcc_chunked(mfcc1, mfcc2, chunk_size=50):
    _, length1 = mfcc1.shape
    _, length2 = mfcc2.shape

    # Calculate the number of chunks
    num_chunks = length2 - chunk_size + 1

    # Initialize an array to store similarity scores for each chunk
    similarities = np.zeros(num_chunks)

    # Slide the window over the larger spectrogram and calculate similarity for each chunk
    for i in range(num_chunks):
        chunk = mfcc2[:, i:i + chunk_size]
        similarities[i] = np.max([compare_mfcc(mfcc1[:, j:j + chunk_size], chunk) for j in range(length1 - chunk_size + 1)])

    # Aggregate similarity scores
    overall_similarity = np.max(similarities)

    return overall_similarity

def preprocess_database(dataset_directory):
    dataset = []
    file_names = []

    for filename in os.listdir(dataset_directory):
        if filename.endswith(".wav"):
            filepath = os.path.join(dataset_directory, filename)
            mfcc = create_mfcc(filepath)
            dataset.append(mfcc)
            file_names.append(filename)

    return dataset, file_names
def create_constellation(audio, Fs):
    # Parameters
    window_length_seconds = 1
    window_length_samples = int(window_length_seconds * Fs)
    window_length_samples += window_length_samples % 2
    num_peaks = 5
    # Pad the song to divide evenly into windows
    amount_to_pad = window_length_samples - audio.size % window_length_samples
    song_input = np.pad(audio, (0, amount_to_pad))
    # Perform a short time fourier transform
    frequencies, times, stft = signal.stft(
        song_input, Fs, nperseg=window_length_samples, nfft=window_length_samples, return_onesided=True
    )
    constellation_map = []
    for time_idx, window in enumerate(stft.T):
        # Spectrum is by default complex. 
        # We want real values only
        spectrum = abs(window)
        # Find peaks - these correspond to interesting features
        # Note the distance - want an even spread across the spectrum
        peaks, props = signal.find_peaks(spectrum, prominence=0, distance=10)
        # Only want the most prominent peaks
        # With a maximum of 15 per time slice
        n_peaks = min(num_peaks, len(peaks))
        # Get the n_peaks largest peaks from the prominences
        # This is an argpartition
        # Useful explanation: https://kanoki.org/2020/01/14/find-k-smallest-and-largest-values-and-its-indices-in-a-numpy-array/
        largest_peaks = np.argpartition(props["prominences"], -n_peaks)[-n_peaks:]
        for peak in peaks[largest_peaks]:
            frequency = frequencies[peak]
            constellation_map.append([time_idx, frequency])
    return constellation_map




# constellation_map = create_constellation(song, Fs)



def create_hashes(constellation_map, song_id=None):
    hashes = {}
    # Use this for binning - 23_000 is slighlty higher than the maximum
    # frequency that can be stored in the .wav files, 22.05 kHz
    upper_frequency = 23_000 
    frequency_bits = 10
    # Iterate the constellation
    for idx, (time, freq) in enumerate(constellation_map):
        # Iterate the next 100 pairs to produce the combinatorial hashes
        # When we produced the constellation before, it was sorted by time already
        # So this finds the next n points in time (though they might occur at the same time)
        for other_time, other_freq in constellation_map[idx : idx + 100]:
            diff = other_time - time
           
            # If the time difference between the pairs is too small or large
            # ignore this set of pairs
            if diff <= 2 or diff > 5:
                continue
            # Place the frequencies (in Hz) into a 1024 bins
            freq_binned = freq / upper_frequency * (2 ** frequency_bits)
            other_freq_binned = other_freq / upper_frequency * (2 ** frequency_bits)
            # Produce a 32 bit hash
            # Use bit shifting to move the bits to the correct location
            hash = int(freq_binned) | (int(other_freq_binned) << 10) | (int(diff) << 20)
            hashes[hash] = (time, song_id)
    return hashes
# # Quickly investigate some of the hashes produced
# hashes = create_hashes(constellation_map, 0)
# for i, (hash, (time, _)) in enumerate(hashes.items()):
#     if i > 10: 
#         break
#     print(f"Hash {hash} occurred at {time}")



# def save_mfccs_to_pickle(dataset, file_names, pickle_filename):
#     mfcc_dict = {'dataset': dataset, 'file_names': file_names}
#     with open(pickle_filename, 'wb') as f:
#         pickle.dump(mfcc_dict, f)
#     print(f"MFCCs saved to {pickle_filename}")
#
# # Load and preprocess the dataset
# dataset_directory = "Mode2"
# dataset, file_names = preprocess_database(dataset_directory)
#
# # Save MFCCs and file names into a pickle file
pickle_filename = 'mfcc_database.pickle'
# save_mfccs_to_pickle(dataset, file_names, pickle_filename)

# Now, you can load the MFCC database from the pickle file when needed
# For loading the pickle file and comparing the user's MFCC, you can modify your existing code as follows:

def load_mfccs_from_pickle(pickle_filename):
    with open(pickle_filename, 'rb') as f:
        mfcc_dict = pickle.load(f)
    return mfcc_dict['dataset'], mfcc_dict['file_names']

# Load MFCC database from the pickle file
loaded_dataset, loaded_file_names = load_mfccs_from_pickle(pickle_filename)












# songs = glob.glob('Mode2/*wav')
#
# song_name_index = {}
# database: Dict[int, List[Tuple[int, int]]] = {}
#
# # Go through each song, using where they are alphabetically as an id
# for index, filename in enumerate(tqdm(sorted(songs))):
#     song_name_index[index] = filename
#     # Read the song, create a constellation and hashes
#     Fs, audio_input = read(filename)
#     constellation = create_constellation(np.array(audio_input).flatten(), Fs)
#     hashes = create_hashes(constellation, index)
#
#     # For each hash, append it to the list for this hash
#     for hash, time_index_pair in hashes.items():
#         if hash not in database:
#             database[hash] = []
#         database[hash].append(time_index_pair)
# # Dump the database and list of songs as pickles
# with open("database.pickle", 'wb') as db:
#     pickle.dump(database, db, pickle.HIGHEST_PROTOCOL)
# with open("song_index.pickle", 'wb') as songs:
#     pickle.dump(song_name_index, songs, pickle.HIGHEST_PROTOCOL)


database = pickle.load(open('database.pickle', 'rb'))
song_name_index = pickle.load(open("song_index.pickle", "rb"))










def score_hashes_against_database(hashes):
    matches_per_song = {}
    for hash, (sample_time, _) in hashes.items():
        if hash in database:
            matching_occurences = database[hash]
            for source_time, song_index in matching_occurences:
                if song_index not in matches_per_song:
                    matches_per_song[song_index] = []
                matches_per_song[song_index].append((hash, sample_time, source_time))
            

    scores = {}
    for song_index, matches in matches_per_song.items():
        song_scores_by_offset = {}
        for hash, sample_time, source_time in matches:
            delta = source_time - sample_time
            if delta not in song_scores_by_offset:
                song_scores_by_offset[delta] = 0
            
            song_scores_by_offset[delta] += 1
        max = (0, 0)
        for offset, score in song_scores_by_offset.items():
            if score > max[1] :
                
                max = (offset, score)
        
        scores[song_index] = max
    # Sort the scores for the user
    scores = list(sorted(scores.items(), key=lambda x: x[1][1], reverse=True)) 
    
    return scores









def compare_mfcc_and_hashes(user_mfcc, hashes):
    # Scores from MFCC comparison
    mfcc_scores = [(compare_mfcc_chunked(user_mfcc, mfcc), file_name) for mfcc, file_name in zip(loaded_dataset, loaded_file_names)]
    
    # Scores from hashing
    hash_scores = score_hashes_against_database(hashes)
    
    # Dictionary to hold combined scores
    combined_scores = {}

    # Combine scores for each file using a weighted average or any suitable method
    for (mfcc_score, mfcc_file), (hash_score, hash_file) in zip(mfcc_scores, hash_scores):
        # Combine scores here using weighted average or any other method you prefer
        combined_score = (1.3 * mfcc_score*10000) + (0.3 * hash_score )  # Adjust weights as needed

        # Store the combined score along with the file name
        combined_scores[mfcc_file] = combined_score
    
    # Sort and return the combined scores
    sorted_combined_scores = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_combined_scores













duration = 2  # Recording duration in seconds
input_audio, fs = record_audio(3)
#fs, input_audio = read("C:/Users/yi/Downloads/record_1.wav")
wavfile.write("output_file.wav", fs, ((input_audio/np.max(input_audio)) * 32767).astype(np.int16))  # Scaling to int16 before saving
# fs, input_audio = read("Train/GrantMeAccessTrainAwad2.wav")

user_audio = "output_file.wav"
user_mfcc = create_mfcc(user_audio)

similarities = [(compare_mfcc_chunked(user_mfcc, mfcc), file_name) for mfcc, file_name in zip(loaded_dataset, loaded_file_names)]

# Get top matches from MFCC comparison
top_mfcc_matches = sorted(similarities, key=lambda x: x[0], reverse=True)[:5]

# Calculate hashes from user's input
constellation_map = create_constellation(np.array(input_audio).flatten(), fs)
hashes = create_hashes(constellation_map, None)

# Get combined scores of MFCC and hashing
combined_scores = compare_mfcc_and_hashes(user_mfcc, hashes)

# Print or use the combined scores as needed
# print("Top MFCC Matches:")
# for score in top_mfcc_matches:
#     print(score)

print("\nCombined Scores (MFCC + Hashing):")
for file_name, score in combined_scores:
    print(f"{file_name}: Score - {score}")
    
scores=score_hashes_against_database(hashes)
for song_id, score in scores:
        print(f"{song_name_index[song_id]}: Score of {score[1]} at {score[0]}")
        if "Mahmoud" in(song_name_index[song_id]) :
            print("batoot")
        break



# this is going to be used inide the main code so send the users checked from the check box and loop over them so whoever exists inside the file name is the user who spoke otherwise invalid user Access Denied 
# for song_id, score in scores:
#         print(f"{song_name_index[song_id]}: Score of {score[1]} at {score[0]}")
#         if user in(song_name_index[song_id]) :
#             Flaguserexists=True
#             return user
#         break