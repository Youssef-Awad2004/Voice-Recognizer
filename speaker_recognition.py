import os
import numpy as np
import librosa

# The Assignment Is Speaker Recognition

# Audio files folder
AUDIO_FILE_PATH = 'C:/Users/galal/Desktop/CHS study folder/CCE-C Year 2 Term 1/Signal/Signal Project/AudioFiles'

# A dictionary which stores dominant frequency values for each speaker
# The keys are student codes
# ALL Dominant frequency values for each student
dom_freq_vals = {1210003: [], 1210280: [], 1210339: []}
# Dictionary that stores the avg feature vector for each speaker in the traininig phase (10)
feat_vector = {1210003: 0, 1210280: 0, 1210339: 0}
# Dictionary that stores the 5 feature vectors for each speaker in the testing phase
feat_vector_test = {1210003: [], 1210280: [], 1210339: []}
# list of all test vectors for all speakers
test_vector_list = []
# Get the dominant frequencies
for speaker in list(dom_freq_vals.keys()):
    try:
        for i in range(1, 16):
            # Create a string which represents the file path
            #  All files are stored in the format of: <Student_ID>_<Record#>.wav
            file_name = f"{speaker}_{i}.wav"
            file_path = os.path.join(AUDIO_FILE_PATH, file_name)

            # y is a 2d array which stores signal magnitudes of each sample and
            # number of channels in the audio (by default mono)
            # sr is the sampling rate (by default sr=22050)
            y, sr = librosa.load(file_path)

            # Perform FFT to convert time-domain signal to frequency-domain
            fft_result = np.fft.fft(y)
            fft_result_magnitude = np.abs(fft_result)

            # Generate frequency axis in Hz
            frequency_axis = np.fft.fftfreq(len(y), 1/sr)

            # Select positive frequencies and corresponding magnitudes
            positive_frequencies = frequency_axis[:len(y)//2]
            positive_magnitudes = fft_result_magnitude[:len(y)//2]

            # Find the index of the frequency with maximum magnitude (dominant frequency)
            dominant_frequency_index = np.argmax(positive_magnitudes)

            # Get the dominant frequency
            # round to 4 significant digits
            dominant_frequency = round(positive_frequencies[dominant_frequency_index], 4)
            dom_freq_vals[speaker].append(dominant_frequency)

        # Get the average feature vector for the speaker
        # Use only the first 10 values to get calculate feature vector (training)
        # round to 4 significant digits
        feat_vector[speaker] = round(np.average(dom_freq_vals[speaker][:10]), 4)

        # Get the features for the test vectors (last 5 values)
        feat_vector_test[speaker] = dom_freq_vals[speaker][10:]
        # Append the test vectors to the list
        test_vector_list.extend(dom_freq_vals[speaker][10:])
    except FileNotFoundError as e:
        print(f"Error: {e}")

# list of avg feature vectors (training)
feat_vector_list = [feat_vector[val] for val in feat_vector]

# We will recognize the student by taking the test vector and finding the student
# whose feature vector is closest to the test vector

# Dictionary that holds the vectors from test vector list
# to the closest matching speaker
closest_match_vals = {1210003: [], 1210280: [], 1210339: []}
# Loop over the avg test vectors toget the closest matching indices
for index, test in enumerate(test_vector_list):
    closest_value = min(feat_vector_list, key=lambda x: abs(x - test))
    closest_index = feat_vector_list.index(closest_value)
    # Check if there has already been a match
    if closest_index == 0:
        print(f"For test {index + 1} the recognized student was 1210003")
        closest_match_vals[1210003].append(test)
    elif closest_index == 1:
        print(f"For test {index + 1} the recognized student was 1210280")
        closest_match_vals[1210280].append(test)
    elif closest_index == 2:
        print(f"For test {index + 1} the recognized student was 1210339")
        closest_match_vals[1210339].append(test)


## The recognition accuracy for each student will be calculated as follows
# get their number of recognitions and divide it by the number of correct
# recognitions
# In case there were no recognitions the recognition accuracy is zero

# This dictionary stores the recognition accuracy for each student
recog_acc_students = {1210003: 0, 1210280: 0, 1210339: 0}

for speaker in recog_acc_students:
    correct_recognitions = 0
    recognitions = len(closest_match_vals[speaker])
    recog_acc = 0
    for val in closest_match_vals[speaker]:
        if val in feat_vector_test[speaker]:
            correct_recognitions += 1
    if recognitions != 0:
        recog_acc = correct_recognitions / recognitions * 100
        recog_acc = round(recog_acc, 4)
    recog_acc_students[speaker] = recog_acc
    print(f"Recognition accuracy for {speaker}: {recog_acc_students[speaker]}%")

# The recognition accuracy for the program will be:
# average of the recognition accuracies of all students
# rounded to 4 decimal places
overall_acc = round(np.average(list(recog_acc_students.values())), 4)
print(f"Overall recognition accuracy: {overall_acc}%")
