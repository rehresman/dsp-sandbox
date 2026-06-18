import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile

#fs = 48_000
cutoff = 3000
num_taps = 4001

fs, input = wavfile.read('/Volumes/T7/Everything/Samples for Digitakt/DFAM Digit Long.wav')

input = np.astype(input, np.int64)

input_range = np.max(input) - np.min(input)
input = input / input_range

# impulse input
#input = np.zeros(l)
#start = l // 2
#input[start:start+32] = 1.0

# design iir filter
l = input.shape[0]

b,a = signal.butter(20, cutoff/(fs/2), 'low')

iir_frequencies, iir_response = signal.freqz(b, a, worN=8192, fs=fs)

# design fir filter to match iir

iir_gains = np.concatenate((np.abs(iir_response), [0]))
iir_gain_freqs = np.concatenate((np.abs(iir_frequencies), [fs/2]))

h = signal.firwin2(
    numtaps=num_taps,
    freq=iir_gain_freqs,
    gain=iir_gains,
    fs=fs,
)

fir_frequencies, fir_response = signal.freqz(h, worN=8192, fs=fs)

fir_sig = signal.lfilter(h, 1.0, input, axis=0)
iir_sig = signal.lfilter(b, a, input, axis=0)

# create .wav file of results

fir_filename = "fir_output.wav"
iir_filename = "iir_output.wav"
fir_array = fir_sig
iir_array = iir_sig
int16_max_amp = 32767
gain = 0.3
fir_array_int16 = np.int16(fir_array * int16_max_amp * gain)
iir_array_int16 = np.int16(iir_array * int16_max_amp * gain)

wavfile.write(fir_filename, fs, fir_array_int16)
wavfile.write(iir_filename, fs, iir_array_int16)

plt.semilogx(
    fir_frequencies,
    20 * np.log10(np.maximum(np.abs(fir_response), 1e-10)),
    label="FIR Response"
)
plt.semilogx(
    iir_frequencies,
    20 * np.log10(np.maximum(np.abs(iir_response), 1e-10)),
    label="IIR Response"
)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
#plt.ylim(-1,1)
plt.ylim(-60, 5)
plt.grid(True)
plt.show()