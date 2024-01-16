from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Dropout, BatchNormalization, Activation, Add, Flatten, Dense
from tensorflow import keras
from tensorflow.keras.initializers import lecun_normal
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
import pandas as pd
import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from ncps import wirings
from ncps.tf import CfC, LTC
from scipy.signal import stft
from scipy.signal import butter, filtfilt, resample
import pywt
from scipy import signal
import os
import csv

import argparse
# import seaborn as sns


parser = argparse.ArgumentParser(
    prog='Model Name',
    description='What do you want to save your Model as',
    epilog='Name of the model'
)

parser.add_argument('file_name', metavar="file_name", type=str, help='Enter the model name you want to save as')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
parser.add_argument('--initial_learning_rate', type=float, default=0.01, help='Initial learning rate')
parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout rate')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
parser.add_argument('--activation', type=str, default='tanh', help='Activation function')
parser.add_argument('--num_repeats', type=int, default=8, help='Number of times to repeat the samples')
parser.add_argument('--n_channels', type=int, default=0, help='Number of times to repeat the samples')

args = parser.parse_args()

file_name = args.file_name
epochs = args.epochs
initial_learning_rate = args.initial_learning_rate
dr = args.dropout_rate
batch_size = args.batch_size
activation = args.activation
num_repeats = args.num_repeats
n = args.n_channels

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
    print("GPU detected: ", physical_devices)
else:
    print("No GPU detected.")

wiring = wirings.AutoNCP(20, 6)
ncp = LTC(wiring, return_sequences=False)

initializer = None

model = keras.models.Sequential()
model.add(keras.layers.InputLayer(input_shape=(None, 12, 33, 129)))
model.add(keras.layers.ConvLSTM2D(filters=16,kernel_size=(12, 3),padding='valid',strides=(1,2),activation=activation,
        dropout=dr, recurrent_dropout=dr, return_sequences=True, kernel_initializer=initializer))
model.add(keras.layers.TimeDistributed(Flatten()))
model.add(Dropout(dr))
model.add(keras.layers.TimeDistributed(Dense(75, activation=activation, kernel_initializer=initializer)))
model.add(ncp)
model.add(keras.layers.Activation("sigmoid"))

# Filters
def apply_bandpass_filter(ecg_data, fs=500, lowcut=0.5, highcut=40, order=4):
    """
    Applies a bandpass filter to each lead in ECG data.

    Args:
    ecg_data (numpy.ndarray): numpy array of shape [N, 4096, 12], where N is the number of ECG recordings
    fs (float): Sampling frequency in Hz (default: 500 Hz)
    lowcut (float): Lower cutoff frequency in Hz (default: 0.5 Hz)
    highcut (float): Upper cutoff frequency in Hz (default: 40 Hz)
    order (int): Filter order (default: 4)

    Returns:
    numpy.ndarray: a numpy array of shape [N, 4096, 12], containing the denoised ECG data
    """
    nyq = 0.5*fs
    lowcut = lowcut/nyq
    highcut = highcut/nyq

    # Create an empty array to store the denoised ECG data
    denoised_ecg_data = np.zeros_like(ecg_data)

    # Loop through each lead in the ECG data
    for i in range(ecg_data.shape[0]):
        for j in range(ecg_data.shape[2]):
            # Extract the ECG data for the current lead
            lead_data = ecg_data[i, :, j]

            # Design the bandpass filter
            b, a = butter(order,[lowcut,highcut], btype='band')

            # Apply the bandpass filter to the lead data
            denoised_lead_data = filtfilt(b, a, lead_data)

            # Store the denoised lead data in the denoised ECG data array
            denoised_ecg_data[i, :, j] = denoised_lead_data

    return denoised_ecg_data

def filter_ecg_signal(data, wavelet='db4', level=8, fs=500, fc=0.1, order=6):
    """
    Filter ECG signals using wavelet denoising.

    Args:
        data (numpy array): ECG signal data with shape (n_samples, n_samples_per_lead, n_leads).
        wavelet (str, optional): Wavelet type for denoising. Default is 'db4'.
        level (int, optional): Decomposition level for wavelet denoising. Default is 8.
        fs (float, optional): Sampling frequency of ECG signals. Default is 500 Hz.
        fc (float, optional): Cutoff frequency for lowpass filter. Default is 0.1 Hz.
        order (int, optional): Filter order for Butterworth filter. Default is 6.

    Returns:
        numpy array: Filtered ECG signals.
    """
    nyquist = 0.5 * fs
    cutoff = fc / nyquist
    b, a = signal.butter(order, cutoff, btype='lowpass')

    filtered_signals = np.zeros_like(data)

    for n in range(data.shape[0]):
        for i in range(data.shape[2]):
            ecg_signal = data[n, :, i]
            coeffs = pywt.wavedec(ecg_signal, wavelet, level=level)
            cA = coeffs[0]
            filtered_cA = signal.filtfilt(b, a, cA)
            filtered_coeffs = [filtered_cA] + coeffs[1:]
            filtered_signal = pywt.waverec(filtered_coeffs, wavelet)
            filtered_signals[n, :, i] = filtered_signal

    return filtered_signals

# resampling ECG data
def resample_ecg_data(ecg_data, origianl_rate, target_rate, samples):
    """
    Resamples ECG data from 400 Hz to 500 Hz.

    Args:
        ecg_data (np.ndarray): ECG data with shape [N, 4096, 12].

    Returns:
        np.ndarray: Resampled ECG data with shape [N, M, 12], where M is the new number of samples after resampling.
    """
    # Compute the resampling ratio
    resampling_ratio = target_rate / origianl_rate

    # Compute the new number of samples after resampling
    M = int(ecg_data.shape[1] * resampling_ratio)

    # Initialize an array to store the resampled data
    ecg_data_resampled = np.zeros((ecg_data.shape[0], M, ecg_data.shape[2]))

    # Iterate over each channel and resample independently
    for i in range(ecg_data.shape[2]):
        for j in range(ecg_data.shape[0]):
            ecg_data_resampled[j, :, i] = resample(ecg_data[j, :, i], M)
    # Trim the resampled data to the last 4096 samples
    ecg_data_resampled = ecg_data_resampled[:, -samples:, :]
    return ecg_data_resampled

def set_channels_to_zero(ecg_data, n):
    """
    Randomly selects a number of ECG channels to set to zero for each group in the data.

    Args:
    - ecg_data: numpy array of shape (N, 4096, 12) containing ECG data
    - n: maximum number of channels that can be set to zero (up to n-1 channels can be left non-zero)

    Returns:
    - numpy array of shape (N, 4096, 12) with selected channels set to zero for each group
    """

    num_groups = 100
    # Choose number of channels to set to zero (up to n-1)
    num_channels_to_set_zero = n
    group_size = ecg_data.shape[0] // num_groups

    for i in range(num_groups):
        start_idx = i * group_size
        end_idx = start_idx + group_size

        group_data = ecg_data[start_idx:end_idx, :, :]

        # Choose which channels to set to zero
        channels_to_set_zero = np.random.choice(group_data.shape[-1], num_channels_to_set_zero, replace=False)

        # Set selected channels to zero
        ecg_data[start_idx:end_idx, :, channels_to_set_zero] = 0

    # Handle the last group separately to avoid going beyond the shape
    start_idx = num_groups * group_size
    group_data = ecg_data[start_idx:, :, :]

    # Choose which channels to set to zero
    channels_to_set_zero = np.random.choice(group_data.shape[-1], num_channels_to_set_zero, replace=False)

    # Set selected channels to zero
    ecg_data[start_idx:, :, channels_to_set_zero] = 0

    return ecg_data


print('Reading Data')
path_to_hdf5 = 'x.hdf5'
hdf5_dset = 'tracings'
path_to_csv = 'y.csv'
f = h5py.File(path_to_hdf5, "r")
x = f[hdf5_dset][:]

# Read the CSV file
label = pd.read_csv(path_to_csv)[['1dAVb','RBBB','LBBB','AF']]
# Get the column names
columns = label.columns
# Convert label values to np.float32 data type
y = label.values.astype(np.float32)

# print('Resampling X')
print('Band passing X')
x = apply_bandpass_filter(x)
print('Filtering X')
x = filter_ecg_signal(x)
print('Emptying X channels')
x = set_channels_to_zero(x, n)


def STFT_ECG_all_channels(sampling_rate, ecg_data):
    # Define the STFT parameters
    window = 'hann'
    nperseg = int(sampling_rate*0.5)
    noverlap = int(sampling_rate*0.5*0.5)
    number_of_signals = ecg_data.shape[-1]
    Zxx_overall = []
    for k in range(ecg_data.shape[0]):
        Zxx_all = []
        for i in range(number_of_signals):
            channel = i
            arr1 = ecg_data[k, :, i]
            # Compute the STFT
            f, t, Zxx = stft(arr1, fs=sampling_rate, window=window, nperseg=256, noverlap=None)

            Zxx_all.append(np.abs(Zxx))
        Zxx_all = np.array(Zxx_all)
        Zxx_overall.append(Zxx_all)
    Zxx_overall = np.array(Zxx_overall).transpose(0, 2, 3, 1)
    return Zxx_overall

print('Transforming x')
x = STFT_ECG_all_channels(500, x)
x = np.transpose(x,[0,3,2,1]) #x_train = data_size,channels,time,n_freq_bins)
x = x.reshape(-1, 1, x.shape[1], x.shape[2], x.shape[3]).astype(np.float32)
print(x.shape)

csv_logger = CSVLogger('./logs/' + file_name + '/test_logger.csv', separator=',', append=True)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=initial_learning_rate),
    metrics=[keras.metrics.Recall(), keras.metrics.Precision(),
             keras.metrics.BinaryAccuracy(), keras.metrics.FalseNegatives(),
             keras.metrics.AUC(curve='ROC'),
            keras.metrics.AUC(curve='PR')],
    loss= keras.losses.BinaryCrossentropy(from_logits=False)
)

model.summary(line_length=100)


# Loading the model
print("Loading Best Model")
model.load_weights('/model.h5')


y_pred = model.predict(x)[:, [0, 1, 2, 4]]
y_true = y

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score


# set a threshold value of 0.5
threshold = 0.5

# apply the threshold to convert the predicted probabilities to binary values
y_pred_bin = np.where(y_pred >= threshold, 1, 0)

# calculate evaluation metrics
precision = precision_score(y_true, y_pred_bin, average=None)
recall = recall_score(y_true, y_pred_bin, average=None)
f1 = f1_score(y_true, y_pred_bin, average=None)
auroc_scores = roc_auc_score(y_true, y_pred, average=None)

# print evaluation metrics
print("Class:", columns)
print("Precision:", precision)
print("Recall:", recall)
print("F1 score:", f1)
print("AUROC:", auroc_scores)

# Write evaluation metrics to a CSV file
csv_name = './logs/'+ file_name +'/test_metrics_channel' + str(n) + '.csv'
with open(csv_name, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Class', 'Precision', 'Recall', 'F1 Score', 'AUROC'])
    for i in range(len(columns)):
        writer.writerow([columns[i], precision[i], recall[i], f1[i], auroc_scores[i]])

    # Write average metrics row
    writer.writerow(['Average', sum(precision)/len(precision), sum(recall)/len(recall), sum(f1)/len(f1), sum(auroc_scores)/len(auroc_scores)])


# Loading the final model
print("Loading final model")
model.load_weights('/model.h5')

y_pred = model.predict(x)[:, [0, 1, 2, 4]]
y_true = y

print(np.sum(y_true, axis=0))

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score


# set a threshold value of 0.5
threshold = 0.5

# apply the threshold to convert the predicted probabilities to binary values
y_pred_bin = np.where(y_pred >= threshold, 1, 0)

# calculate evaluation metrics
precision = precision_score(y_true, y_pred_bin, average=None)
recall = recall_score(y_true, y_pred_bin, average=None)
f1 = f1_score(y_true, y_pred_bin, average=None)
auroc_scores = roc_auc_score(y_true, y_pred, average=None)

# print evaluation metrics
print("Class:", columns)
print("Precision:", precision)
print("Recall:", recall)
print("F1 score:", f1)
print("AUROC:", auroc_scores)

# Write evaluation metrics to a CSV file
csv_name = './logs/'+ file_name +'/final_test_metrics_channel' + str(n) + '.csv'
with open(csv_name, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Class', 'Precision', 'Recall', 'F1 Score', 'AUROC'])
    for i in range(len(columns)):
        writer.writerow([columns[i], precision[i], recall[i], f1[i], auroc_scores[i]])

    # Write average metrics row
    writer.writerow(['Average', sum(precision)/len(precision), sum(recall)/len(recall), sum(f1)/len(f1), sum(auroc_scores)/len(auroc_scores)])
