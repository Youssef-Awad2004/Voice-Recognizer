import os
import librosa
import pickle
import numpy as np
import pyaudio
from scipy.io import wavfile






def record_audio(duration):
    CHUNK = 1024  # Number of frames per buffer
    FORMAT = pyaudio.paFloat32  # Audio format (32-bit floating-point)
    CHANNELS = 1  # Number of audio channels (1 for mono, 2 for stereo)
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







def create_mfcc(audio_file, target_shape=None):
    y, sr = librosa.load(audio_file)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    # Resize the MFCC if a target shape is specified
    if target_shape is not None:
        mfcc = librosa.util.fix_length(mfcc, size=target_shape[1], mode='constant', constant_values=0)

    return mfcc

def compare_mfcc(mfcc1, mfcc2):
    # Use Euclidean distance for comparison
    distance = np.linalg.norm(mfcc1 - mfcc2)

    # Calculate similarity as the inverse of the distance
    similarity = 1 / (1 + distance)

    return similarity

def preprocess_database(dataset_directory, target_shape=None):
    dataset = []
    file_names = []

    for filename in os.listdir(dataset_directory):
        if filename.endswith(".wav"):
            filepath = os.path.join(dataset_directory, filename)
            mfcc = create_mfcc(filepath, target_shape)
            dataset.append(mfcc)
            file_names.append(filename)

    return dataset, file_names

input_audio, fs = record_audio(3)
wavfile.write("output_file.wav", fs, (input_audio * 32767).astype(np.int16))  # Scaling to int16 before saving

user_mfcc = create_mfcc("output_file.wav", target_shape=(13, 128))  # Adjust the target shape as needed

# Load and preprocess the dataset
dataset_directory = "PleaseWork"
dataset, file_names = preprocess_database(dataset_directory, target_shape=user_mfcc.shape)

# Compare user's MFCC with each MFCC in the dataset
similarities = [(compare_mfcc(user_mfcc, mfcc), file_name) for mfcc, file_name in zip(dataset, file_names)]

# Get top matches
top_matches = sorted(similarities, key=lambda x: x[0], reverse=True)[:3]

print(top_matches)
