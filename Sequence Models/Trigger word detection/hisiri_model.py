

from keras.callbacks import ModelCheckpoint
# 引入Tensorboard
from keras.callbacks import TensorBoard
from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Activation, Dropout, Input, Masking, TimeDistributed, LSTM, Conv1D
from keras.layers import GRU, Bidirectional, BatchNormalization, Reshape
from keras.optimizers import Adam
from audio_data import graph_spectrogram
import numpy as np



Y = np.load('train_dir/Y.npy')
Y_test = np.load('train_dir/Y_test.npy')
# number of frequncies in fourier decomposition
freq_n = 101
# number of samples in the audio clip
sample_n = 1998

Y_FIX = Y.swapaxes(1,2)
Y_test_FIX = Y_test.swapaxes(1,2)

train_dir = "train_dir"

# Y = Y.reshape(1000,1375,4)
# Y_test = Y_test.reshape(100,1375,4)

def load_training_data(train_dir,num_train=1000,num_test=100):
    X = np.zeros((num_train,freq_n,sample_n))
    X_test = np.zeros((num_test,freq_n,sample_n))

    for i in range(num_train):
        X[i,:,:] = graph_spectrogram(train_dir + "/train" + str(i) + ".wav")

    for i in range(num_test):
        X_test[i,:,:] = graph_spectrogram(train_dir + "/traintest" + str(i) + ".wav")


    return X.reshape(num_train,sample_n,freq_n), X_test.reshape(num_test,sample_n,freq_n)
X, X_test = load_training_data(train_dir)

#
X       = np.load('train_dir/X.npy')
X_test  = np.load('train_dir/X_test.npy')

Ty = 1375 # The number of time steps in the output of the model

def model(input_shape):
    """
    Function creating the model's graph in Keras.
    Argument:
    input_shape -- shape of the model's input data (using Keras conventions)
    Returns:
    model -- Keras model instance
    """

    X_input = Input(shape = input_shape)

    # Step 1: CONV layer (≈4 lines)  flavor:  kernel_size = 15 in coursera
    X = Conv1D(196, kernel_size=624, strides=1)(X_input)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Dropout(0.8)(X)

    # Step 2: First GRU Layer (≈4 lines)
    X = GRU(units = 128, return_sequences = True)(X)
    X = Dropout(0.8)(X)
    X = BatchNormalization()(X)

    # Step 3: Second GRU Layer (≈4 lines)
    X = GRU(units = 128, return_sequences = True)(X)
    X = Dropout(0.8)(X)
    X = BatchNormalization()(X)
    X = Dropout(0.8)(X)

    # Step 4: Time-distributed dense layer (≈1 line)
    X = TimeDistributed(Dense(4, activation = "sigmoid"))(X) # time distributed  (sigmoid)

    model = Model(inputs = X_input, outputs = X)
    return model

# num_train = 1000
# Y_NEW = np.zeros((num_train,Ty,4))
# for i in range(num_train):
#     spec = Y[i, :, :]
#     Y_NEW[i, :, :] = spec.reshape(Ty, 4, order='F')



model = model(input_shape = (sample_n, freq_n))
model.summary()

opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.01)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=["accuracy"])

# model.fit(X, Y, batch_size = 5, epochs=10)
model.fit(X, Y_FIX, batch_size = 5, epochs=10)

model.save("audio_wake_model.h5")

# loss, acc = model.evaluate(X_test, Y_test)
# print("Dev set accuracy = ", acc)
#
#
# def detect_triggerword(filename):
#     plt.subplot(2, 1, 1)
#
#     x = graph_spectrogram(filename)
#     # the spectogram outputs (freqs, Tx) and we want (Tx, freqs) to input into the model
#     x  = x.swapaxes(0,1)
#     x = np.expand_dims(x, axis=0)
#     predictions = model.predict(x)
#
#     plt.subplot(2, 1, 2)
#     plt.plot(predictions[0,:,0])
#     plt.ylabel('probability')
#     plt.show()
#     return predictions
