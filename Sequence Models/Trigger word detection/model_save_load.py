import numpy as np
import os
import json
from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Activation, Dropout, Input, Masking, TimeDistributed, LSTM, Conv1D
from keras.layers import GRU, Bidirectional, BatchNormalization, Reshape
from keras.optimizers import Adam
from keras.models import model_from_json

# 改变交互式的当前路径
os.getcwd()
os.chdir('G:/GitRepos/deep-learning-coursera/Sequence Models/Trigger word detection/')

model = load_model('./models/tr_model.h5')
json_string = model.to_json()
print(json_string)

model2 = load_model('./models/fan_audio_wake_model.h5')
json_string2 = model2.to_json()
print(json_string2)
