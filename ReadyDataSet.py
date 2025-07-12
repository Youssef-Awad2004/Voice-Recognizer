import os
import librosa
import pickle


def create_spectrogram(audio_file):
    y, sr = librosa.load(audio_file)
    spect = librosa.feature.melspectrogram(y=y, sr=sr)
    return spect



dataset_directory = "Train"

# Assuming each file in the directory is a WAV file
dataset = []

for filename in os.listdir(dataset_directory):
    if filename.endswith(".wav"):
        filepath = os.path.join(dataset_directory, filename)
        spect = create_spectrogram(filepath)
        dataset.append(spect)

with open('dataset.pkl', 'wb') as f:
    pickle.dump(dataset, f)





