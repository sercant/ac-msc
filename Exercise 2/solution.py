
# import f0lib
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy import signal
from scipy import stats
import scipy
from python_speech_features import mfcc
from scipy.signal import lfilter
from f0Lib import getF0


def resample(x, source_fs, target_fs):
    return signal.resample(x, int(len(x) * target_fs / source_fs))

# BEGIN TASK 0 #

fig, ax = plt.subplots(nrows=4, ncols=1)

# 1. Load the 'speech_sample'
sample_speech = sio.loadmat('lab2_data.mat')

# 2. Declare the source sampling frequency, and the target sampling frequency.
#    2.1 Source sampling frequency
fs_source = 48000

#    2.2 Target sampling frequency
# Target frequency
fs_target = 11025

# #. Downsample the speech sample
x = sample_speech['speech_sample'].ravel()
x0 = resample(x, fs_source, fs_target)

# Visualize the downsampled speech signal.
ax[0].plot(x0)

# END TASK 0 #

# BEGIN TASK 1 #

# BEGIN TASK 1.1 #

# 1. Pre-emphasize your resampled signal.
#    1.1 Define the polynomial of your fitler
#        filter coefficients b, which is the numerator
#        filter coefficients a, which is the denominator
coef = 0.98
b = np.array([1., -coef], x0.dtype)
a = np.array([1.], x0.dtype)
x1 = signal.lfilter(b, a, x0)

ax[0].plot(x1)

# 2. Extract the mfcc coefficient by callying the mfcc() function
# remeber to set the pre-emphasize argument to 0 since the signal has been pre-emphasized.
frameLen = int(2 ** np.floor(np.log2(0.03 * fs_target)))
mfccContour = mfcc(x1,
                   samplerate=fs_target,
                   winlen=float(frameLen)/fs_target,
                   winstep=float(frameLen)/(2*fs_target),
                   numcep=12,
                   preemph=0)


# 3. Plot the 12 mfcc contours
ax[1].plot(mfccContour)

# 4. Calculate the mean for each contour.
mean_mfccContour = np.mean(mfccContour, axis=tuple(range(mfccContour.ndim - 1)))

# END TASK 1.1 #

# BEGIN TASK 1.2 #

# 1. Define a hamming window
#    1.1 Calculate the window length, which the number of frames within 0.01s
sample_count_per_frame = int(0.01 * fs_target)
window_length = int(np.ceil(len(x0) / sample_count_per_frame))        # number of non-overlapping _full_ frames

#    1.2 Define the hamming window using signal.hamming()
hamming_window = signal.windows.hamming(window_length)

# 2. Calculate the short time energy (STE) contour by convolve the hamming window and the squared signal,
#    using the scipy.signal.convolve() function
x12 = signal.convolve(x0 ** 2, hamming_window)

# 3. Clip half window of frames from both the beginning and end of the STE contour
clip = int(np.ceil(sample_count_per_frame / 2))
x12 = x12[clip:-clip]

# 4. Visualize the final STE contour.
ax[2].plot(x12)

# 5. Calculate the 5 distribution parameter feature the of STE contour
mean_ste = np.mean(x12)
std_ste = np.std(x12)
ten_perc_ste = np.percentile(x12, 10)
ninty_perc_ste = np.percentile(x12, 90)
kurtosis_ste = stats.kurtosis(x12)

# END TASK 1.2 #

# BEGIN TASK 1.3 #

# 1. Extract the F0 contour
F0, strength, T_ind, wflag = getF0(x0, fs_target)

# 2. Visualize the F0 contour
ax[3].plot(F0)

# 3. Calculate these distribution parameter features
mean_F0 = np.mean(F0)
std_F0 = np.std(F0)
ten_perc_F0 = np.percentile(F0, 10)
ninty_perc_F0 = np.percentile(F0, 90)
kurtosis_F0 = stats.kurtosis(F0)

# END TASK 1.3 #

# BEGIN TASK 1.4 #


# END TASK 1.4 #

# END TASK 1 #

fig.tight_layout()
plt.show()
