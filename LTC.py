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
import pickle

import argparse
# import seaborn as sns



parser = argparse.ArgumentParser(
    prog='Model Name',
    description='What do you want to save your Model as',
    epilog='Name of the model'
)

parser.add_argument('file_name', metavar="file_name", type=str, help='Enter the model name you want to save as')
parser.add_argument('--epochs', type=int, default=300, help='Number of epochs')
parser.add_argument('--initial_learning_rate', type=float, default=0.01, help='Initial Learning rate')
parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout rate')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
parser.add_argument('--activation', type=str, default='relu', help='Activation function')
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

if activation == 'selu':
    initializer = keras.initializers.lecun_normal(seed=42)
if activation == 'tanh':
    initializer = 'glorot_uniform'
else:
    initializer = 'he_normal'

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
# path_to_hdf5 = '/mnt/data13_16T/jim/ECG_data/Brazil/filtered_data20000.hdf5'
hdf5_dset = 'tracings'
path_to_csv = 'y.csv'
# path_to_csv = '/mnt/data13_16T/jim/ECG_data/Brazil/filtered_annotations20000.csv'
f = h5py.File(path_to_hdf5, "r")
x = f[hdf5_dset][:]

# Read the CSV file
label = pd.read_csv(path_to_csv)[['1dAVb','RBBB','LBBB','SB','AF','ST']]
# Get the column names
columns = label.columns
# Convert label values to np.float32 data type
y = label.values.astype(np.float32)

print('Resampling X')
x = resample_ecg_data(x, 400, 500, 4096)
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



x_train, x_val, y_train, y_val = train_test_split(x, y, test_size = 0.2, random_state=42, shuffle=True)

# Resampling Data
# Assuming you have y_train and x_train as numpy arrays
indices = np.where(y_train.sum(axis=1) > 0)[0]
sampled_y_train = y_train[indices]
sampled_x_train = x_train[indices]

# Repeat the samples
repeated_y_train = np.repeat(sampled_y_train, num_repeats, axis=0)
repeated_x_train = np.repeat(sampled_x_train, num_repeats, axis=0)

# Concatenate the repeated samples with the original training dataset
y_train = np.concatenate((y_train, repeated_y_train), axis=0)
x_train = np.concatenate((x_train, repeated_x_train), axis=0)

print(x_train.shape, y_train.shape, type(x_train), type(y_train))
print(x_val.shape, y_val.shape, type(x_val), type(y_val))


if not os.path.exists('./logs/' + file_name):
    os.makedirs('./logs/' + file_name)



# Convert x_val and y_val to numpy arrays if they are not already
x_val = np.array(x_val)
y_val = np.array(y_val)

# Save x_val and y_val as numpy arrays
np.save(file_name + '/x_val.npy', x_val)
np.save(file_name + '/y_val.npy', y_val)





csv_logger = CSVLogger('./logs/' + file_name + '/train_logger.csv', separator=',', append=True)

# Define the checkpoint path and filename
checkpoint_path = './logs/'+ file_name +'/' + file_name + '.h5'

# Define the ModelCheckpoint callback
best_checkpoint_callback = ModelCheckpoint(checkpoint_path,
                                      monitor='val_auc_1',
                                      save_best_only=True,
                                      save_weights_only=True,
                                      mode='max',
                                      verbose=1)

final_checkpoint_path = './logs/'+ file_name +'/final_' + file_name + '.h5'
final_checkpoint_callback = ModelCheckpoint(final_checkpoint_path,
                                            save_weights_only=True,
                                            verbose=1)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=initial_learning_rate),
    metrics=[keras.metrics.Recall(), keras.metrics.Precision(),
             keras.metrics.BinaryAccuracy(), keras.metrics.FalseNegatives(),
             keras.metrics.AUC(curve='ROC'),
            keras.metrics.AUC(curve='PR')],
    loss= keras.losses.BinaryCrossentropy(from_logits=False)
)

model.summary(line_length=100)

print(model.predict(x_val))

model.evaluate(x_val, y_val)

history = model.fit(
    x=x_train, y=y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_val, y_val),
    callbacks=[csv_logger, best_checkpoint_callback, final_checkpoint_callback], shuffle=True,
)




# Loading the model
print("Loading Best Model")
model.load_weights('/model.h5')


y_pred = model.predict(x_val)
y_true = y_val


from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score


# set a threshold value of 0.5
threshold = 0.5

# apply the threshold to convert the predicted probabilities to binary values
y_pred_bin = np.where(y_pred >= threshold, 1, 0)

# calculate evaluation metrics
# tn, fp, fn, tp = confusion_matrix(y_true, y_pred_bin).ravel()
precision = precision_score(y_true, y_pred_bin, average=None)
recall = recall_score(y_true, y_pred_bin, average=None)
f1 = f1_score(y_true, y_pred_bin, average=None)
auroc_scores = roc_auc_score(y_true, y_pred, average=None)
# specificity = tn / (tn + fp)

# print evaluation metrics
print("Class:", columns)
print("Precision:", precision)
print("Recall:", recall)
print("F1 score:", f1)
print("AUROC:", auroc_scores)

# Write evaluation metrics to a CSV file
csv_name = './logs/'+ file_name +'/evaluation_metrics.csv'
with open(csv_name, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Class', 'Precision', 'Recall', 'F1 Score', 'AUROC'])
    for i in range(len(columns)):
        writer.writerow([columns[i], precision[i], recall[i], f1[i], auroc_scores[i]])

    # Write average metrics row
    writer.writerow(['Average', sum(precision)/len(precision), sum(recall)/len(recall), sum(f1)/len(f1), sum(auroc_scores)/len(auroc_scores)])


# Save the model to a file
filename = './logs/'+ file_name +'/' + file_name + '.pkl'
pickle.dump(model, open(filename, 'wb'))


# Loading the final model
print("Loading final model")
model.load_weights('/model.h5')

y_pred = model.predict(x_val)
y_true = y_val

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
csv_name = './logs/'+ file_name +'/final_evaluation_metrics.csv'
with open(csv_name, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Class', 'Precision', 'Recall', 'F1 Score', 'AUROC'])
    for i in range(len(columns)):
        writer.writerow([columns[i], precision[i], recall[i], f1[i], auroc_scores[i]])

    # Write average metrics row
    writer.writerow(['Average', sum(precision)/len(precision), sum(recall)/len(recall), sum(f1)/len(f1), sum(auroc_scores)/len(auroc_scores)])

# Save the model to a file
filename = './logs/'+ file_name +'/final_' + file_name + '.pkl'
pickle.dump(model, open(filename, 'wb'))
