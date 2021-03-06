

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

# get_ipython().magic('matplotlib inline')

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

def load_raw_audio(train_dir):
    ones = []
    twos = []
    threes = []
    backgrounds = []
    negatives = []

    for filename in os.listdir(train_dir+"/one"):
        if filename.endswith("wav"):
            one = AudioSegment.from_wav(train_dir+"/one/"+filename)
            ones.append(one)
    for filename in os.listdir(train_dir+"/two"):
        if filename.endswith("wav"):
            two = AudioSegment.from_wav(train_dir+"/two/"+filename)
            twos.append(two)
    for filename in os.listdir(train_dir+"/three"):
        if filename.endswith("wav"):
            three = AudioSegment.from_wav(train_dir+"/three/"+filename)
            threes.append(three)

    for filename in os.listdir(train_dir+"/background"):
        if filename.endswith("wav"):
            background = AudioSegment.from_wav(train_dir+"/background/"+filename)
            backgrounds.append(background[:10000])
    for filename in os.listdir(train_dir+"/negative"):
        if filename.endswith("wav"):
            negative = AudioSegment.from_wav(train_dir+"/negative/"+filename)
            negatives.append(negative)
    return ones, twos, threes, negatives, backgrounds



ones, twos, threes, negatives, backgrounds = load_raw_audio('train_dir')



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
        if ((segment_end >= previous_start) and (segment_start <= previous_end)):
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
    # picking new segment_time at random until it doesn't overlap. (≈ 2 lines)
    while is_overlapping(segment_time,previous_segments):
        segment_time = get_random_time_segment(len(audio_clip))
    previous_segments.append(segment_time)
    # Superpose audio segment and background
    new_background = background.overlay(audio_clip, position = segment_time[0])
    return new_background, segment_time


def insert_ones(y, segment_end_ms, label):
    """
    Update the label vector y. The labels of the 50 output steps strictly after the end of the segment
    should be set to 1. By strictly we mean that the label of segment_end_y should be 0 while, the
    50 followinf labels should be ones.


    Arguments:
    y -- numpy array of shape (4, Ty), the labels of the training example
    segment_end_ms -- the end time of the segment in ms
    label -- class of sound, i.e one/two/three/negative

    Returns:
    y -- updated labels
    """

    # duration of the background (in terms of spectrogram time-steps)
    segment_end_y = int(segment_end_ms * Ty / 10000.0)
    for i in range(segment_end_y+1, segment_end_y + 51):
        if i < Ty:
            if(label == 'one'):
                y[0,i] = 1
            elif(label == 'two'):
                y[1,i] = 1
            elif(label == 'three'):
                y[2,i] = 1
            elif(label == 'negative'):
                y[3,i] = 1
    return y

def create_single_training_example(background, ones, twos, threes, negatives, train_dir,suffix=None):
    """
    Creates a training example with a given background, activates, and negatives.

    Arguments:
    background -- a 10 second background audio recording
    ones -- a list of audio segments of the word "one"
    twos -- a list of audio segments of the word "two"
    threes -- a list of audio segments of the word "three"
    negatives -- a list of audio segments of random words that are not "one/two/three"
    suffix -- a string to add at the end of the filename

    Returns:
    x -- the spectrogram of the training example
    y -- the label at each time step of the spectrogram
    """

    # Make background quieter
    background = background - 20
    y = np.zeros((4, Ty))
    previous_segments = []

    number_of_ones = np.random.randint(0, 2)
    number_of_twos = np.random.randint(0, 2)
    number_of_threes = np.random.randint(0, 2)
    random_indices_one = np.random.randint(2350, size=number_of_ones) #min(len(ones),len(twos),len(threes)) = 2350
    random_indices_two = np.random.randint(2350, size=number_of_twos)
    random_indices_three = np.random.randint(2350, size=number_of_threes)
    random_ones = [ones[i] for i in random_indices_one]
    random_twos = [twos[i] for i in random_indices_two]
    random_threes = [threes[i] for i in random_indices_three]

    for random_one in random_ones:
        # Insert the audio clip on the background
        background, segment_time = insert_audio_clip(background, random_one, previous_segments)
        # Retrieve segment_start and segment_end from segment_time
        segment_start, segment_end = segment_time
        # Insert labels in "y"
        y = insert_ones(y, segment_end_ms=segment_end,label='one')
    for random_two in random_twos:
        # Insert the audio clip on the background
        background, segment_time = insert_audio_clip(background, random_two, previous_segments)
        # Retrieve segment_start and segment_end from segment_time
        segment_start, segment_end = segment_time
        # Insert labels in "y"
        y = insert_ones(y, segment_end_ms=segment_end,label='two')
    for random_three in random_threes:
        # Insert the audio clip on the background
        background, segment_time = insert_audio_clip(background, random_three, previous_segments)
        # Retrieve segment_start and segment_end from segment_time
        segment_start, segment_end = segment_time
        # Insert labels in "y"
        y = insert_ones(y, segment_end_ms=segment_end,label='three')



    number_of_negatives = np.random.randint(0, 2)
    random_indices_negative = np.random.randint(1561, size=number_of_negatives) #len(negatives) = 1561
    random_negatives = [negatives[i] for i in random_indices_negative]


    for random_negative in random_negatives:
        # Insert the audio clip on the background
        background, segment_time = insert_audio_clip(background, random_negative, previous_segments)
        segment_start, segment_end = segment_time
        y = insert_ones(y, segment_end_ms=segment_end,label='negative')


    # Standardize the volume of the audio clip
    background = match_target_amplitude(background, -20.0)

    # Export new training example
    file_handle = background.export(train_dir + suffix + ".wav", format="wav")


    return y

Ty = 1375 # The number of time steps in the output of the model

#  Y shape shold be (1375, 4) not (4,1275)
def generate_training_set(backgrounds,ones,twos,threes,negatives,num_train=1000,num_test=100):
    Y = np.zeros((num_train,4,Ty))
    Y_test = np.zeros((num_test,4,Ty))
    train_dir = "train_dir/train/"
    test_dir = "train_dir/test/"
    for i in range(num_train):
        j = np.random.randint(0,6) #number of background tracks = 3
        y = create_single_training_example(backgrounds[j], ones, twos, threes, negatives,train_dir, str(i))
        Y[i] = y

    for i in range(num_test):
        j=np.random.randint(0,6)
        y = create_single_training_example(backgrounds[j], ones, twos, threes, negatives, test_dir, str(i))
        Y_test[i] = y

    np.save("train_dir/Y.npy",Y)
    np.save("train_dir/Y_test.npy",Y_test)
    return 0


generate_training_set(backgrounds,ones,twos,threes,negatives)

num_train =1000
freq_n = 101
sample_n = 1998

# os.getcwd()
spec = graph_spectrogram(train_dir + str(0) + ".wav")  # (101, 1998)
# spec.shape
X = np.zeros((num_train, sample_n,freq_n))  # (1000, 1998, 101)
# X.shape
X[0,:,:] = spec.reshape(sample_n,freq_n,order='F')





def convert_wavs_to_npy(train_dir,save_name="X",num_train=1000,freq_n=101,sample_n=1998,):
    X = np.zeros((num_train, sample_n, freq_n))  # (1000, 1998, 101)
    for i in range(num_train):
        # x = graph_spectrogram(train_dir + str(0) + ".wav")  # (101, 1998)
        x = graph_spectrogram(train_dir +"test" +str(0) + ".wav")  # (101, 1998)
        X[i, :, :] = x.swapaxes(0,1)
    out_name = train_dir + save_name + ".npy"
    np.save(out_name, X)

train_dir = './train_dir/train/'
convert_wavs_to_npy(train_dir,"X")
test_dir = './train_dir/test/'
convert_wavs_to_npy(test_dir,"X_test",num_train=100)
# A = np.arange(6)
# np.reshape(A,(3,2))
# A.reshape(2,3,order='F')