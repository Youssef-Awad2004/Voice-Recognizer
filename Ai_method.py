import os
import numpy as np
import librosa
from hmmlearn import hmm
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Function for extracting MFCC features
def extract_mfcc(audio_data, sr):
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
    return mfccs.T  # Transpose to have time along the rows

# Directory containing your WAV files
data_directory = 'Mode1'

# Collect all WAV files in the directory
wav_files = [os.path.join(data_directory, file) for file in os.listdir(data_directory) if file.endswith(".wav")]

# Load audio data and extract features
data = []
labels = []
lengths = []

for wav_file in wav_files:
    audio_data, _ = librosa.load(wav_file, sr=None)
    mfcc_features = extract_mfcc(audio_data, _)
    if 'Code' in wav_file:
        label = 0  # You need to define how to get labels based on file names or other criteria
    elif 'Access' in wav_file:
        label = 1
    elif 'Enter' in wav_file:
        label = 2
    data.append(mfcc_features)
    labels.append(label)
    lengths.append(len(mfcc_features))

data = np.vstack(data)
labels = np.array(labels)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test, lengths_train, lengths_test = train_test_split(
    data, labels, lengths, test_size=0.2, random_state=42
)

# Number of states and mixtures
num_states = 2
num_mix = 2

# Train Gaussian Mixture Models
gmm_models = []
for state in range(num_states):
    gmm = GaussianMixture(n_components=num_mix)
    state_data = X_train[y_train == state]
    gmm.fit(state_data)
    gmm_models.append(gmm)

# Initialize HMM using GMMs
startprob = np.ones(num_states) / num_states
transmat = np.ones((num_states, num_states)) / num_states

hmm_model = hmm.GMMHMM(n_components=num_states, n_mix=num_mix, covariance_type='full')
hmm_model.startprob_ = startprob
hmm_model.transmat_ = transmat
hmm_model._initialize(X_train, lengths_train)

# Train HMM using GMM parameters
hmm_model.fit(X_train, lengths_train)

# Evaluate the model on the test set
y_pred = []
for i, sample in enumerate(X_test):
    log_likelihood = np.array([gmm.score([sample]) for gmm in gmm_models]).sum()
    state = hmm_model.predict([sample], lengths=[lengths_test[i]])[0]
    y_pred.append(state)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Save the model if accuracy is greater than 80%
if accuracy > 0.8:
    joblib.dump(hmm_model, 'speech_recognition_model.joblib')
    print("Model saved successfully!")
