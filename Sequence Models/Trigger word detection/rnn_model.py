import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

import pyaudio

from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Activation, Dropout, Input, Masking, TimeDistributed, LSTM, Conv1D
from keras.layers import GRU, Bidirectional, BatchNormalization, Reshape
from keras.optimizers import Adam

Tx = 5511  # The number of time steps input to the model from the spectrogram
n_freq = 101  # Number of frequencies input to the model at each time step of the spectrogram
Ty = 1375  # The number of time steps in the output of our model

def model(input_shape):
    """
    Function creating the model's graph in Keras.

    Argument:
    input_shape -- shape of the model's input data (using Keras conventions)

    Returns:
    model -- Keras model instance
    """
    X_input = Input(shape=input_shape)

    # Step 1: CONV layer (≈4 lines)
    X = Conv1D(256, kernel_size=15, strides=4)(X_input)  # CONV1D

    X = BatchNormalization()(X)  # Batch normalization
    X = Activation('relu')(X)  # ReLu activation
    X = Dropout(0.8)(X)  # dropout (use 0.8)

    # Step 2: First GRU Layer (≈4 lines)
    X = GRU(units=128, return_sequences=True)(X)  # GRU (use 128 units and return the sequences)
    X = Dropout(0.8)(X)  # dropout (use 0.8)
    X = BatchNormalization()(X)  # Batch normalization

    # Step 3: Second GRU Layer (≈4 lines)
    X = GRU(units=128, return_sequences=True)(X)  # GRU (use 128 units and return the sequences)
    X = Dropout(0.8)(X)  # dropout (use 0.8)
    X = BatchNormalization()(X)  # Batch normalization
    X = Dropout(0.8)(X)  # dropout (use 0.8)

    # Step 4: Time-distributed dense layer (≈1 line)
    X = TimeDistributed(Dense(1, activation="sigmoid"))(X)  # time distributed  (sigmoid)

    model = Model(inputs=X_input, outputs=X)
    return model

tbCallBack = TensorBoard(log_dir='/data/train_dir/trigger_word',  # log 目录
                 histogram_freq=0,  # 按照何等频率（epoch）来计算直方图，0为不计算
#                  batch_size=32,     # 用多大量的数据计算直方图
                 write_graph=True,  # 是否存储网络结构图
                 write_grads=True, # 是否可视化梯度直方图
                 write_images=True,# 是否可视化参数
                 embeddings_freq=0,
                 embeddings_layer_names=None,
                 embeddings_metadata=None)

# def detect_triggerword(filename):
#     plt.subplot(2, 1, 1)
#
#     x = graph_spectrogram(filename)
#     # the spectogram outputs (freqs, Tx) and we want (Tx, freqs) to input into the model
#     x = x.swapaxes(0, 1)
#     x = np.expand_dims(x, axis=0)
#     predictions = model.predict(x)
#
#     plt.subplot(2, 1, 2)
#     plt.plot(predictions[0, :, 0])
#     plt.ylabel('probability')
#     plt.show()
#     return predictions




if __name__ == '__main__':
    model = model(input_shape=(Tx, n_freq))
    model.summary()

    opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.01)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])

    # model.fit(X, Y, batch_size=5, epochs=1)
    X = np.load("./X.npy")
    Y = np.load("./Y.npy")

    X_dev = np.load("./X_test.npy")
    Y_dev = np.load("./Y_test.npy")

    model.fit(X, Y, batch_size=5, epochs=100, callbacks=[tbCallBack])
    model.save("./models/fan_trigger_word_model.h5")

    # Dev    set    accuracy = 0.9359999895095825
    loss, acc = model.evaluate(X_dev, Y_dev)
    print("Dev set accuracy = ", acc)

    # filename = "./raw_data/dev/1.wav"
    # filename = "./test_dir/test3.wav"
    # filename = "./train_dir/train476.wav"
    # prediction = detect_triggerword(filename)

#     model compare
#     model = load_model('./models/tr_model.h5')
#     model = load_model('./models/fan_audio_wake_model_V2.h5')

#######



