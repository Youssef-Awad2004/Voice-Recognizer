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


import librosa
import numpy as np
import pyaudio

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

# all_peaks, props = signal.find_peaks(transform_y)
# peaks, props = signal.find_peaks(transform_y, prominence=0, distance=100)
# n_peaks = 10
# # Get the n_peaks largest peaks from the prominences
# # This is an argpartition
# # Useful explanation: https://kanoki.org/2020/01/14/find-k-smallest-and-largest-values-and-its-indices-in-a-numpy-array/
# largest_peaks_indices = np.argpartition(props["prominences"], -n_peaks)[-n_peaks:]

# largest_peaks = peaks[largest_peaks_indices]
# plt.plot(transform_x, transform_y, label="Spectrum")
# plt.scatter(transform_x[largest_peaks], transform_y[largest_peaks], color="r", zorder=10, label="Constrained Peaks")
# plt.xlim(0, 2000)
# plt.show()




# # Some parameters
# window_length_seconds = 0.001
# window_length_samples = int(window_length_seconds * Fs)
# window_length_samples += window_length_samples % 2
# # Perform a short time fourier transform
# # frequencies and times are references for plotting/analysis later
# # the stft is a NxM matrix
# frequencies, times, stft = signal.stft(
#     song, Fs, nperseg=window_length_samples,
#     nfft=window_length_samples, return_onesided=True
# )
# print(stft.shape)





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
        for other_time, other_freq in constellation_map[idx : idx + 1000]:
            diff = other_time - time
            # If the time difference between the pairs is too small or large
            # ignore this set of pairs
            if diff <= 1 or diff > 6:
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








songs = glob.glob('Mode2/*.wav')

song_name_index = {}
database: Dict[int, List[Tuple[int, int]]] = {}

# Go through each song, using where they are alphabetically as an id
for index, filename in enumerate(tqdm(sorted(songs))):
    song_name_index[index] = filename
    # Read the song, create a constellation and hashes
    Fs, audio_input = read(filename)
    constellation = create_constellation(np.array(audio_input).flatten(), Fs)
    hashes = create_hashes(constellation, index)

    # For each hash, append it to the list for this hash
    for hash, time_index_pair in hashes.items():
        if hash not in database:
            database[hash] = []
        database[hash].append(time_index_pair)
# Dump the database and list of songs as pickles
with open("database.pickle", 'wb') as db:
    pickle.dump(database, db, pickle.HIGHEST_PROTOCOL)
with open("song_index.pickle", 'wb') as songs:
    pickle.dump(song_name_index, songs, pickle.HIGHEST_PROTOCOL)


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
            if score > max[1]:
                max = (offset, score)
        
        scores[song_index] = max
    # Sort the scores for the user
    scores = list(sorted(scores.items(), key=lambda x: x[1][1], reverse=True)) 
    
    return scores


duration = 2  # Recording duration in seconds
input_audio, fs = record_audio(3)
# fs, input_audio = read("Train/GrantMeAccessTrainAwad2.wav")
constellation_map=create_constellation(np.array(input_audio).flatten(), fs)
hashes = create_hashes(constellation_map, None)
scores=score_hashes_against_database(hashes)
for song_id, score in scores:
        print(f"{song_name_index[song_id]}: Score of {score[1]} at {score[0]}")
