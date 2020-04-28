import numpy as np
from scipy.io import wavfile
import os
import librosa
import simpleaudio as sa
import wave, struct
import time

class Data:
  def __init__(self, path, window, skip):
    self.sr = 8000 # sampling rate
    self.window = window # size of each data
    self.skip = skip # the gap between two row of data
    try:
      self.input, self.target = self.load_data()
    except:
      self.input, self.target = self.dump_data(path)
    print('Data size:', self.input.shape)

  def dump_data(self, path):
    input, target = [], []
    for r, d, f in os.walk(path):
      for file in f:
        filename = path + file
        print('Processing:', filename)
        i, t = self.process_file(filename)
        input += i
        target += t
    input = np.array(input)
    target = np.array(target)
    np.save('input', input)
    np.save('target', target)
    return input, target

  def load_data(self):
    return np.load('input.npy'), np.load('target.npy')

  def process_file(self, filename):
    input, target = [], []
    sig, sr = librosa.load(filename, sr=self.sr, mono=True, dtype=np.float32)
    for i in range(0, len(sig)-self.window-self.window*2//3, self.skip):
      input.append(sig[i: i+self.window])
      target.append(sig[i+self.window: i+self.window+self.window*2//3])
    # self.play_audio(input[0])
    return input, target

  def play_audio(self, data):
    play_obj = sa.play_buffer(data, 1, 4, self.sr)
    play_obj.wait_done()

  def write_file(self, filename, data):
    wavfile.write(filename, self.sr, data)

  def ZC(self, x):
    """add zero crossing, return (N, window)"""
    a2 = np.sign(x)
    change2 = ((np.roll(a2, 1, axis=1) - a2)!=0).astype(int)
    return change2

  def MAV(self, x):
    """compute mean absolute value, return (N, 1)"""
    return np.mean(np.abs(x), axis=1)

  def SSC(self, x):
    """detect slope change, return (N, window-1)"""
    a = np.sign(np.diff(x, axis=1))
    change = ((np.roll(a, 1, axis=1) - a) != 0).astype(int)
    return change

  def WL(self, x):
    """waveform length feature extraction, return (N, 1)"""
    a = np.sum(np.diff(x, axis=1), axis=1)
    return a

  def STD(self, x):
    """standard deviation of the channels, return (N, 1)"""
    return x.std(axis=1)

  def RMS(self, x):
    """Root mean squared, return (N, 1)"""
    return (np.mean(x ** 2, axis=1)) ** (1 / 2)

  def get_train_data(self):
    zc = self.ZC(self.input) # (N, window)
    mav = self.MAV(self.input)[:, np.newaxis] # (N, 1)
    ssc = self.SSC(self.input) # (N, window-1)
    wl = self.WL(self.input)[:, np.newaxis] # (N, 1)
    std = self.STD(self.input)[:, np.newaxis] # (N, 1)
    rms = self.RMS(self.input)[:, np.newaxis] # (N, 1)
    oned_feature = np.hstack((mav, wl, std, rms))
    return (self.input, zc, ssc, oned_feature), self.target




