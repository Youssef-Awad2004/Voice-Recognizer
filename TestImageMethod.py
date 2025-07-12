import os
import librosa
import numpy as np
import librosa
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



duration = 2  # Recording duration in seconds
input_audio, fs = record_audio(2)
wavfile.write("output_file.wav", fs, ((input_audio/np.max(input_audio)) * 32767).astype(np.int16))  # Scaling to int16 before saving



user_audio = "output_file.wav"
user_mfcc = create_mfcc(user_audio)

# Load and preprocess the dataset
dataset_directory = "PleaseWork"
dataset, file_names = preprocess_database(dataset_directory)

# Compare user's MFCC with each MFCC in the dataset using a sliding window
similarities = [(compare_mfcc_chunked(user_mfcc, mfcc), file_name) for mfcc, file_name in zip(dataset, file_names)]

# Get top matches
top_matches = sorted(similarities, key=lambda x: x[0], reverse=True)[:3]

print(top_matches)
