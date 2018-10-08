
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


# BEGIN TASK 0 #

fig, ax = plt.subplots(nrows=4, ncols=1)

# 1. Load the 'speech_sample'
exerciseData = sio.loadmat('lab2_data.mat')
speechSample = exerciseData['speech_sample'].reshape(-1)
# 2. Declare the source sampling frequency, and the target sampling frequency.
#    2.1 Source sampling frequency
fs_source = 48000

#    2.2 Target sampling frequency
# Target frequency
fs_down = 11025

speechLength = len(speechSample) / fs_source

# #. Downsample the speech sample
speech_resampled = signal.resample(speechSample, int(
    np.round(speechLength * fs_down)))

# 4. Visualize the downsampled speech signal.
#    4.1 Creating the corresponding time vector, whose length is same to the length to the given signal.
#        You can use np.linspace() function to perform this. For example

#    4.2 Plot your result
# ax[0].plot(speechSample)
ax[0].plot(np.linspace(0, speechLength, len(speech_resampled)),
           speech_resampled, label='Downsampled Signal')

# END TASK 0 #

# BEGIN TASK 1 #

# BEGIN TASK 1.1 #

# 1. Pre-emphasize your resampled signal.
#    1.1 Define the polynomial of your fitler
#        filter coefficients b, which is the numerator
#        filter coefficients a, which is the denominator
coef = 0.98
b = np.array([1., -coef], speech_resampled.dtype)
a = np.array([1.], speech_resampled.dtype)
preEmphasizedSample = signal.lfilter(b, a, speech_resampled)

# ax[0].plot(np.linspace(0, speechLength, len(speech_resampled)),
#            preEmphasizedSample, label='Pre-emphasized Signal')

# 2. Extract the mfcc coefficient by callying the mfcc() function
# remeber to set the pre-emphasize argument to 0 since the signal has been pre-emphasized.
frameLen = int(2 ** np.floor(np.log2(0.03 * fs_down)))
mfccContour = mfcc(preEmphasizedSample,
                   samplerate=fs_down,
                   winlen=float(frameLen)/fs_down,
                   winstep=float(frameLen)/(2*fs_down),
                   numcep=12,
                   preemph=0)

# 3. Plot the 12 mfcc contours
for k in range(0, len(mfccContour)):
    ax[1].plot(np.linspace(0, speechLength,
                           len(mfccContour[k])), mfccContour[k])

# 4. Calculate the mean for each contour.
mean_mfccContour = np.mean(
    mfccContour, axis=tuple(range(mfccContour.ndim - 1)))

# Why do we need to pre-emphasize the speech signal before computing the MFCC feature?

# The first step is to apply a pre-emphasis filter on the signal to amplify the high
# frequencies. A pre-emphasis filter is useful in several ways: (1) balance the frequency
# spectrum since high frequencies usually have smaller magnitudes compared to lower
# frequencies, (2) avoid numerical problems during the Fourier transform operation
# and (3) may also improve the Signal-to-Noise Ratio (SNR).
# Pre-emphasis often used for loudness equalization

# END TASK 1.1 #

# BEGIN TASK 1.2 #

# 1. Define a hamming window
#    1.1 Calculate the window length, which the number of frames within 0.01s
# number of non-overlapping _full_ frames
window_length = int(np.round(0.01 * fs_down))

#    1.2 Define the hamming window using signal.hamming()
hamming_window = signal.windows.hamming(window_length)

# 2. Calculate the short time energy (STE) contour by convolve the hamming window and the squared signal,
#    using the scipy.signal.convolve() function
x12 = signal.convolve(speech_resampled ** 2, hamming_window)

# 3. Clip half window of frames from both the beginning and end of the STE contour
clip = int(np.ceil(window_length / 2))
x12 = x12[clip:-clip]

# 4. Visualize the final STE contour.
ax[2].plot(np.linspace(0, speechLength, len(x12)), x12)

# 5. Calculate the 5 distribution parameter feature the of STE contour
mean_ste = np.mean(x12)
std_ste = np.std(x12)
ten_perc_ste = np.percentile(x12, 10)
ninty_perc_ste = np.percentile(x12, 90)
kurtosis_ste = stats.kurtosis(x12)

# END TASK 1.2 #

# BEGIN TASK 1.3 #

# 1. Extract the f0 contour
f0, _, T_ind, _ = getF0(speech_resampled, fs_down)

# 2. Visualize the f0 contour
ax[3].plot(T_ind[:, 0] / fs_down, f0)

# 3. Calculate these distribution parameter features
mean_f0 = np.mean(f0)
std_f0 = np.std(f0)
ten_perc_f0 = np.percentile(f0, 10)
ninty_perc_f0 = np.percentile(f0, 90)
kurtosis_f0 = stats.kurtosis(f0)

# END TASK 1.3 #

# BEGIN TASK 1.4 #

# 1. Segmenting the voiced and unvoiced speech segements.
#    1.1 Example on extracting voiced segment lengths
framesInd_voiced = np.where(f0 > 0)[0]
diff = framesInd_voiced[1:] - framesInd_voiced[0:-1]
voiceToUnviceInd = np.where(diff > 1)[0]
voice_seg_num = len(voiceToUnviceInd) + 1
voice_seg_lengths = np.zeros(voice_seg_num)
tmp = framesInd_voiced[0]

for i in range(voice_seg_num - 1):
    voice_seg_lengths[i] = framesInd_voiced[voiceToUnviceInd[i]] - tmp + 1
    tmp = framesInd_voiced[voiceToUnviceInd[i] + 1]

voice_seg_lengths[-1] = framesInd_voiced[-1] - tmp + 1

#####################################################################
###################################################################
#    1.2 Extract unvoiced segment lengths.
framesInd_unvoiced = np.where(f0 == 0)[0]
diff = framesInd_unvoiced[1:] - framesInd_unvoiced[0:-1]
ind = np.where(diff > 1)[0]
unvoiced_seg_num = len(ind) + 1
unvoiced_seg_lengths = np.zeros(unvoiced_seg_num)
tmp = framesInd_unvoiced[0]

for i in range(unvoiced_seg_num - 1):
    unvoiced_seg_lengths[i] = framesInd_unvoiced[ind[i]] - tmp + 1
    tmp = framesInd_unvoiced[ind[i] + 1]

unvoiced_seg_lengths[-1] = framesInd_unvoiced[-1] - tmp + 1

# 2. Calculate the means and SDs of both Voiced and Unvoiced segment lengths
mean_voiced = np.mean(voice_seg_lengths)
mean_unvoiced = np.mean(unvoiced_seg_lengths)

std_voiced = np.std(voice_seg_lengths)
std_unvoiced = np.std(unvoiced_seg_lengths)

# 3. Calculate the voicing ratio.
voicing_ratio = np.sum(voice_seg_lengths) / np.sum(unvoiced_seg_lengths)

# END TASK 1.4 #

# START TASK 1.5 #

# 1. Print the 12 MFCC coefficients
print('--- 12 MFCC coefficients ---')
for k in range(0, len(mean_mfccContour)):
    print('{}: {}'.format(k, mean_mfccContour[k]))
print('')

# 2. Print the distribution paremeter feature of the STE contour
print('--- Distribution paremeters of the STE contour ---')
print(
    'mean_ste: ', mean_ste, '\n'
    'std_ste: ', std_ste, '\n'
    'ten_perc_ste: ', ten_perc_ste, '\n'
    'ninty_perc_ste: ', ninty_perc_ste, '\n'
    'kurtosis_ste: ', kurtosis_ste, '\n'
)

# 3. Print the distribution parameter feature of the F0 contour
print('--- Distribution paremeters of the F0 contour ---')
print(
    'mean_f0: ', mean_f0, '\n'
    'std_f0: ', std_f0, '\n'
    'ten_perc_f0: ', ten_perc_f0, '\n'
    'ninty_perc_f0: ', ninty_perc_f0, '\n'
    'kurtosis_f0: ', kurtosis_f0, '\n'
)

# 3. Print the 5 prodosic features
print('--- 5 prodosic features ---')
print(
    'mean_voiced: ', mean_voiced, '\n'
    'mean_unvoiced: ', mean_unvoiced, '\n'
    'std_voiced: ', std_voiced, '\n'
    'std_unvoiced: ', std_unvoiced, '\n'
    'voicing_ratio: ', voicing_ratio, '\n'
)
# END TASK 1.4 #

# END TASK 1 #

fig.tight_layout()
plt.show()
