
import numpy as np
from pydub import AudioSegment
import random
import sys
import io
import os
import glob
import IPython
import matplotlib.pyplot as plt
from scipy.io import wavfile

Tx = 5511  # The number of time steps input to the model from the spectrogram
n_freq = 101  # Number of frequencies input to the model at each time step of the spectrogram
Ty = 1375  # The number of time steps in the output of our model

# Calculate and plot spectrogram for a wav audio file
def graph_spectrogram(wav_file):
    rate, data = get_wav_info(wav_file)
    nfft = 200 # Length of each window segment
    fs = 8000 # Sampling frequencies
    noverlap = 120 # Overlap between windows
    nchannels = data.ndim
    if nchannels == 1:
        pxx, freqs, bins, im = plt.specgram(data, nfft, fs, noverlap = noverlap)
    elif nchannels == 2:
        pxx, freqs, bins, im = plt.specgram(data[:,0], nfft, fs, noverlap = noverlap)
    return pxx

# Load a wav file
def get_wav_info(wav_file):
    rate, data = wavfile.read(wav_file)
    return rate, data

# Used to standardize volume of audio clip
def match_target_amplitude(sound, target_dBFS):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)

# Load raw audio files for speech synthesis
def load_raw_audio(train_dir):
    activates = []
    backgrounds = []
    negatives = []
    for filename in os.listdir(train_dir +"/activates"):
        if filename.endswith("wav"):
            activate = AudioSegment.from_wav(train_dir +"/activates/"+filename)
            activates.append(activate)
    for filename in os.listdir(train_dir +"/backgrounds"):
        if filename.endswith("wav"):
            background = AudioSegment.from_wav(train_dir +"/backgrounds/"+filename)
            backgrounds.append(background)
    for filename in os.listdir(train_dir +"/negatives"):
        if filename.endswith("wav"):
            negative = AudioSegment.from_wav(train_dir +"/negatives/"+filename)
            negatives.append(negative)
    return activates, negatives, backgrounds


def get_random_time_segment(segment_ms):
    """
    Gets a random time segment of duration segment_ms in a 10,000 ms audio clip.

    Arguments:
    segment_ms -- the duration of the audio clip in ms ("ms" stands for "milliseconds")

    Returns:
    segment_time -- a tuple of (segment_start, segment_end) in ms
    """
    segment_start = np.random.randint(low=0, high=10000-segment_ms)   # Make sure segment doesn't run past the 10sec background
    segment_end = segment_start + segment_ms - 1
    return (segment_start, segment_end)

def is_overlapping(segment_time, previous_segments):
    """
    Checks if the time of a segment overlaps with the times of existing segments.

    Arguments:
    segment_time -- a tuple of (segment_start, segment_end) for the new segment
    previous_segments -- a list of tuples of (segment_start, segment_end) for the existing segments

    Returns:
    True if the time segment overlaps with any of the existing segments, False otherwise
    """
    segment_start, segment_end = segment_time
    overlap = False
    for previous_start, previous_end in previous_segments:
        if previous_start <= segment_start <= previous_end or previous_start <= segment_end <= previous_end:
            overlap = True
    return overlap

def insert_audio_clip(background, audio_clip, previous_segments):
    """
    Insert a new audio segment over the background noise at a random time step, ensuring that the
    audio segment does not overlap with existing segments.

    Arguments:
    background -- a 10 second background audio recording.
    audio_clip -- the audio clip to be inserted/overlaid.
    previous_segments -- times where audio segments have already been placed

    Returns:
    new_background -- the updated background audio
    """
    # Get the duration of the audio clip in ms
    segment_ms = len(audio_clip)
    segment_time = get_random_time_segment(segment_ms)
    while is_overlapping(segment_time, previous_segments):
        segment_time = get_random_time_segment(segment_ms)
    previous_segments.append(segment_time)
    new_background = background.overlay(audio_clip, position=segment_time[0])
    return new_background, segment_time


def insert_ones(y, segment_end_ms):
    """
    Update the label vector y. The labels of the 50 output steps strictly after the end of the segment
    should be set to 1. By strictly we mean that the label of segment_end_y should be 0 while, the
    50 followinf labels should be ones.


    Arguments:
    y -- numpy array of shape (1, Ty), the labels of the training example
    segment_end_ms -- the end time of the segment in ms

    Returns:
    y -- updated labels
    """
    segment_end_y = int(segment_end_ms * Ty / 10000.0)

    for i in range(segment_end_y + 1, segment_end_y + 51):
        if i < Ty:
            y[0, i] = 1
    return y




def create_single_training_example(background, activates, negatives, train_dir,suffix=None):
    """
    Creates a training example with a given background, activates, and negatives.

    Arguments:
    background -- a 10 second background audio recording
    activates -- a list of audio segments of the word "activate"
    negatives -- a list of audio segments of random words that are not "activate"

    Returns:
    x -- the spectrogram of the training example
    y -- the label at each time step of the spectrogram
    """
    # np.random.seed(18)
    background = background - 20

    y = np.zeros((1, Ty))
    previous_segments = []

    number_of_activates = np.random.randint(0, 5)
    random_indices = np.random.randint(len(activates), size=number_of_activates)

    random_activates = [activates[i] for i in random_indices]
    for random_activate in random_activates:
        background, segment_time = insert_audio_clip(background, random_activate, previous_segments)
        segment_start, segment_end = segment_time
        y = insert_ones(y, segment_end)

    number_of_negatives = np.random.randint(0, 3)
    random_indices = np.random.randint(len(negatives), size=number_of_negatives)
    random_negatives = [negatives[i] for i in random_indices]

    for random_negative in random_negatives:
        background, _ = insert_audio_clip(background, random_negative, previous_segments)

    background = match_target_amplitude(background, -20.0)

    # Export new training example
    file_handle = background.export(train_dir + suffix +  ".wav", format="wav")
    print(train_dir + suffix +  ".wav" + " File  was saved.")
    x = graph_spectrogram(train_dir + suffix +  ".wav")
    return x, y

#  Y shape shold be (1375, 4) not (4,1275)
def generate_training_set(backgrounds,activates,negatives,num_train=1000,train_dir="train_dir",suffix="train"):
    backgrounds_len = len(backgrounds)

    X = np.zeros((num_train,Tx,n_freq))
    Y = np.zeros((num_train,Ty,1))

    # train_dir = "train_dir/train/"
    # test_dir = "train_dir/test/"
    for i in range(num_train):
        j = np.random.randint(0,backgrounds_len) #number of background tracks = 3
        x,y = create_single_training_example(backgrounds[j], activates, negatives,train_dir, suffix + str(i))
        X[i] = x.swapaxes(0,1)
        Y[i] = y.swapaxes(0,1)

    # np.save(train_dir + "/Y.npy",Y)
    # np.save(train_dir + "/X.npy", X)
    return X,Y




if __name__ == '__main__':

    # os.getcwd()
    # os.chdir('G:/GitRepos/deep-learning-coursera/Sequence Models/Trigger word detection/')

    activates, negatives, backgrounds = load_raw_audio('./raw_data')

    # np.random.seed(5)

    X,Y = generate_training_set(backgrounds,activates,negatives,1000,"./train_dir/","train")



    #  To Do Save To npy
    np.save("./Y.npy", Y)
    np.save("./X.npy", X)

    X_test, Y_test = generate_training_set(backgrounds, activates, negatives, 10, "./test_dir/", "test")

    np.save("./Y_test.npy", Y_test)
    np.save("./X_test.npy", X_test)

    print("done")



