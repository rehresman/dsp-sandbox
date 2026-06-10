from scipy import signal
from scipy.io import wavfile
import scipy.fft
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# setup 

fs = 48000

# generate test signal

sig_seconds = 5
sig_size = fs * sig_seconds
dry = np.random.normal(0,1,sig_size)
dry = dry / np.sqrt(np.mean(dry **2)) # normalize RMS to 1
t = np.linspace(0,sig_seconds,sig_size, endpoint= False)
dry = signal.sawtooth(np.pi*220*t)

# dsp

nyquist = fs / 2
order = 2
cutoff = 800

b,a = signal.butter(order, cutoff, btype="lowpass", fs=fs)


# implement dsp

wet = signal.lfilter(b,a, dry)

# create .wav file of results

wav_filename = "output.wav"
wav_array = wet
int16_max_amp = 32767
gain = 0.3
wav_array_int16 = np.int16(wav_array * int16_max_amp * gain)

wavfile.write(wav_filename, fs, wav_array_int16)

# plot results

# FFT
'''
N = fs
window = np.hanning(N)

segment = wet[:N] * window
yf = scipy.fft.rfft(segment)
xf = scipy.fft.rfftfreq(N, 1/fs)

# normalize fft size and window gain
window_gain = np.sum(window) / N
mag = np.abs(yf) / (N * window_gain)
mag[1:-1] *= 2

mag_db = 20*np.log10(mag + 1e-12)

plt.figure(figsize=(10,8))
plt.semilogx(xf, mag_db)
plt.xlim(20, fs/2)
plt.ylim(-60,10)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.grid(True, which="both")
plt.show()
'''
# PSD
# power spectral density
f_dry, Pxx_dry = signal.welch(dry, fs=fs, nperseg = 4096)
f_wet, Pxx_wet = signal.welch(wet, fs=fs, nperseg = 4096)

plt.figure(figsize=(10,5))
plt.semilogx(f_dry, 10 * np.log10(Pxx_dry + 1e-20))
plt.semilogx(f_wet, 10 * np.log10(Pxx_wet + 1e-20))
plt.xlim(0, fs / 2)
plt.ylim(-100, 0)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power Spectral Density (dB/Hz)")
plt.grid(True, which="both")
plt.legend()
ax = plt.gca()
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax.xaxis.set_minor_formatter(ticker.NullFormatter())
ax.set_xticks([20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000])

plt.show()
