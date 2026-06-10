from scipy import signal
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt

# setup 

fs = 48000

# generate test signal (white noise)

sig_seconds = 5
sig_size = fs * sig_seconds
sig = np.random.normal(0,1,sig_size)
#t = np.linspace(0,1,sig_size)
#sig = signal.sawtooth(2*np.pi*220*t)

# dsp

nyquist = fs / 2
numtaps = 93
cutoff = 100
width = 100

taps = signal.firwin(numtaps, cutoff, width=width, fs=fs)

freq, response = signal.freqz(taps)

# implement dsp

out = signal.convolve(sig, taps, mode='same')

# create .wav file of results

wav_filename = "output.wav"
wav_array = out
int16_max_amp = 32767
gain = 0.3
wav_array_int16 = np.int16(wav_array * int16_max_amp * gain)

wavfile.write(wav_filename, fs, wav_array_int16)

# plot results

print(response[:100])

plt.figure(figsize=(10,8))
plt.semilogy(0.5*fs*freq/np.pi, np.abs(response))

plt.show()