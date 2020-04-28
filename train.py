from data import Data
import numpy as np
from model import BILSTM
import pickle
import time

def play_signal(signal: Data):
  for i in range(signal.input.shape[0]):
    print("input")
    print(signal.input[i, :])
    signal.play_audio(signal.input[i, :])
    time.sleep(1)
    print("target")
    print(signal.target[i, :])
    signal.play_audio(signal.target[i, :])
    time.sleep(2)

# def generate(window):
#   global signal
#   start_sequence = signal.input[0,:]
#   for i in range(33):
#     input = start_sequence[-window:]
#     output = model.predict(input)
#     np.array(start_sequence, output)
#   signal.play_audio(start_sequence)


window = 68000
skip = window // 6
signal = Data('./data/', window, skip)
(input, zc, ssc, oned_feature), target = signal.get_train_data()
input = input[:,:,np.newaxis]
zc = zc[:,:,np.newaxis]
ssc = ssc[:,:,np.newaxis]
# print(np.min(input))
# print(np.max(input))
# play_signal(signal)

model = BILSTM()
# # model.load_model()
model.train(input, zc, ssc, oned_feature, target)
save = input("Save model?")
# previous test accuracy: 91
if int(save) == 1:
    model.save_model()
#
# generate(window)




