# -------------------- Imports -----------------------------------------------------------------------------------------
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
import sys
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential, load_model, Model
from keras.layers import LSTM, Dense, Dropout, Input, MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.optimizers import Adam
import scipy as sp

# -------------------- Coding ------------------------------------------------------------------------------------------

VISU_PLOTS = False  # Visualization of Data ON/OFF
VISU_CORR = False  # Visualization of Correlation matrices ON/OFF
RUN_MODEL = True  # Model training and testing ON/OFF
EVAL_MODEL = True  # Evaluation through loading model

# TODO: change for different experiments
MODEL = True  # True == RNN / False == SAN
MODE = True  # True == Separately / False == generalized data
PATIENT_TBD = 1  # change for different patients

# Parameter changes
k = 72  # Number of past BG values
PH = 4  # Prediction horizon

# folder naming:
if MODEL:
    modelname = "RNN"
else:
    modelname = "SAN"
if MODE:
    modename = "separately"
else:
    modename = "generalized"

folder = modelname + "_" + modename + "_k-" + str(k) + "_PH-" + str(PH) + "_patient-" + str(PATIENT_TBD)

print(f"\n >>> Running now: {folder} <<<\n")

# Load Data
path = "Ohio Data"

raw_train_data = []
raw_test_data = []

fileHistory = []
dataFolders = os.listdir(path)
dataFolders.sort()
for year in dataFolders:
    fileHistory.append(year)
    train_path = os.path.join(path, year, "train")
    test_path = os.path.join(path, year, "test")

    for root, dirs, files in os.walk(train_path):
        files.sort()
        for file in files:
            fileHistory.append(file)
            if file.endswith('.csv'):  # Check if the file is a CSV file
                file_path = os.path.join(root, file)  # full file path
                data = pd.read_csv(file_path)
                raw_train_data.append(data)

    for root, dirs, files in os.walk(test_path):
        files.sort()
        for file in files:
            fileHistory.append(file)
            if file.endswith('.csv'):  # Check if the file is a CSV file
                file_path = os.path.join(root, file)  # full file path
                data = pd.read_csv(file_path)
                raw_test_data.append(data)

print(f"\n >>> Printing file loading history {fileHistory}\n")


# The following are the columns of the csv files:
#   5minute_intervals_timestamp:    Timestamp in five-minute intervals
#   missing_cbg:                    Missing Continuous Blood Glucose
#   cbg:                            Continuous Blood Glucose
#   finger:                         Finger-stick blood glucose readings
#   basal:                          Basal insulin rate
#   hr:                             Heart rate
#   gsr:                            Galvanic Skin Response, a measure of skin conductance
#   carbInput:                      Carbohydrate intake estimation
#   bolus:                          Bolus insulin injection


def prepareData(patient_data, normalizeParam=None):
    print("Preparing data")

    if normalizeParam is None:
        normalizeParam = ['cbg', 'finger', 'basal', 'hr', 'gsr', 'carbInput', 'bolus']

    param_max = dict.fromkeys(normalizeParam,1e-10)
    preparedData = []

    for patientIdx in range(len(patient_data)):
        preparedData.append(patient_data[patientIdx].copy(deep=True))

        # interpolate missing CBG values not with cubic spline interpolation but pchip
        preparedData[patientIdx]['cbg'] = preparedData[patientIdx]['cbg'].interpolate(method='pchip')
        preparedData[patientIdx]['cbg'] = preparedData[patientIdx]['cbg'].fillna(np.mean(preparedData[patientIdx]['cbg']))

        # replace all other NaN by 0
        preparedData[patientIdx] = preparedData[patientIdx].fillna(0)

        # save the highest value for normalization
        for param in normalizeParam:
            param_max[param] = max(max(preparedData[patientIdx][param]),param_max[param])

    # normalization
    for patientIdx in range(len(patient_data)):
        for param in normalizeParam:
            preparedData[patientIdx][param] = preparedData[patientIdx][param] / param_max[param]  # """

    raw_train_set = patient_data[:12]
    raw_test_set = patient_data[12:]

    train_set = preparedData[:12]
    test_set = preparedData[12:]

    if VISU_PLOTS:
        figsize = (20*1.9*6.4/4.8, 20)
        plt.figure(figsize=figsize)
        for idx in range(len(raw_train_set)):
            plt.subplot(3, 4, idx + 1)
            plt.plot(raw_train_set[idx]['cbg'][500:1000])
            plt.xlabel('Timestamps in 5 Minute Intervals')
            plt.ylabel('CBG / mg/dL')
            plt.title('TRAIN_SET: CBG over Time, patient' + str(idx))
        plt.savefig('./img/raw_train_set_WINDOW.png')

        plt.figure(figsize=figsize)
        for idx in range(len(train_set)):
            plt.subplot(3, 4, idx + 1)
            plt.plot(train_set[idx]['cbg'][500:1000])
            plt.xlabel('Timestamps in 5 Minute Intervals')
            plt.ylabel('CBG')
            plt.title('TRAIN_SET: Interpolated CBG over Time, patient' + str(idx))
        plt.savefig('./img/train_set_WINDOW.png')

        plt.figure(figsize=figsize)
        for idx in range(len(raw_test_set)):
            plt.subplot(3, 4, idx + 1)
            plt.plot(raw_test_set[idx]['cbg'][500:1000])
            plt.xlabel('Timestamps in 5 Minute Intervals')
            plt.ylabel('CBG / mg/dL')
            plt.title('TEST_SET: CBG over Time, patient' + str(idx))
        plt.savefig('./img/raw_test_set_WINDOW.png')

        plt.figure(figsize=figsize)
        for idx in range(len(test_set)):
            plt.subplot(3, 4, idx + 1)
            plt.plot(test_set[idx]['cbg'][500:1000])
            plt.xlabel('Timestamps in 5 Minute Intervals')
            plt.ylabel('CBG')
            plt.title('TEST_SET: Interpolated CBG over Time, patient' + str(idx))
        plt.savefig('./img/test_set_WINDOW.png')

    # split into train and test data set again
    return train_set, test_set, param_max


def correlationMatrix(patient_data, prediction_horizon, target='cbg'):

    corr_matrix_cbg_next = []

    for patientIdx in range(len(patient_data)):

        # Create a new column with the next row of the target column
        patient_data[patientIdx][target + '_next'] = patient_data[patientIdx][target].shift(-prediction_horizon)

        # Calculate the correlation matrix
        # extract only cbg_next from the correlation
        corr_matrix_cbg_next.append(patient_data[patientIdx].drop(labels=['5minute_intervals_timestamp','missing_cbg'],
                                                                  axis=1).corr()[target + '_next'])
        # all correlation matrix
        corr_matrix = patient_data[patientIdx].drop(labels=['5minute_intervals_timestamp','missing_cbg'],
                                                    axis=1).corr()

        # Display the correlation values
        if VISU_CORR:
            print(f"Correlation with {target}_next:")
            print(corr_matrix)

    mean_corr_matrix_cbg_next = pd.DataFrame(np.mean(corr_matrix_cbg_next, axis=0),
                                             index=corr_matrix_cbg_next[1].keys(),
                                             columns=[corr_matrix_cbg_next[1].keys()[-1]])
    print(f"Mean of all patient_data cbg_next correlation:")
    print(mean_corr_matrix_cbg_next)

    # combine all correlation matrices in a list
    corr_matrix_dataframe = pd.DataFrame(corr_matrix_cbg_next).drop(labels=['cbg_next'], axis=1)
    if VISU_CORR:
        # create a boxplot of the correlation values
        corr_matrix_dataframe.plot(kind='box')
        plt.xlabel('Correlation with ' + target + '_next')
        plt.ylabel('Correlation value')
        plt.title('Correlation of ' + target + f"_next (stepsize:{prediction_horizon}) with other parameters ")
        plt.xticks(range(1, len(corr_matrix_dataframe.columns) + 1), corr_matrix_dataframe.columns)
        plt.savefig(f"./img/correlation_matrix_stepOfNext_{prediction_horizon}.png")

    return mean_corr_matrix_cbg_next


def build_san(input_shape, num_head, d_model, ff_dim, dropout):
    inputs = Input(shape=input_shape)

    # Self-Attention layers
    self_attention_1 = MultiHeadAttention(num_heads=num_head, key_dim=d_model)(inputs, inputs)
    x = LayerNormalization(epsilon=1e-6)(inputs + self_attention_1)

    self_attention_2 = MultiHeadAttention(num_heads=num_head, key_dim=d_model*2)(x, x)
    x = LayerNormalization(epsilon=1e-6)(x + self_attention_2)

    # Global Average Pooling to reduce temporal dimension
    x = GlobalAveragePooling1D()(x)

    # Feed Forward
    x = Dense(ff_dim, activation='relu')(x)
    x = Dropout(dropout)(x)

    # output
    outputs = Dense(1, activation='sigmoid')(x)

    # Model
    model = Model(inputs=inputs, outputs=outputs, name='san')

    return model


# Prepare sequences for the model
def sequences(patient_data, window, horizon, X, Y):
    for idx in range(len(patient_data) - window - horizon + 1):
        cbg = patient_data['cbg'].values[idx:idx + window]
        basal = patient_data['basal'].values[idx:idx + window]
        carb = patient_data['carbInput'].values[idx:idx + window]
        bolus = patient_data['bolus'].values[idx:idx + window]
        X.append([cbg, basal, carb, bolus])
        Y.append(patient_data['cbg'].values[idx + window - 1 + horizon])
    return X, Y


# prepare data
train_data, test_data, param_max = prepareData(raw_train_data + raw_test_data)

# create correlation matrix
correlationMatrix(train_data, PH)  # with future cbg
correlationMatrix(train_data, 0)  # with current cbg

# initialize lists
X_train = []
X_test = []
Y_train = []
Y_test = []

if MODE:  # separately
    X_train, Y_train = sequences(train_data[PATIENT_TBD], k, PH, X_train, Y_train)
    X_test, Y_test = sequences(test_data[PATIENT_TBD], k, PH, X_test, Y_test)
else:  # generalized
    for patient in range(len(train_data)):
        if patient == PATIENT_TBD:
            X_test, Y_test = sequences(test_data[patient], k, PH, X_test, Y_test)
        else:
            X_train, Y_train = sequences(train_data[patient], k, PH, X_train, Y_train)

# Reshaping
X_train = np.array(X_train).reshape(-1, k, 4)
X_test = np.array(X_test).reshape(-1, k, 4)
Y_train = np.array(Y_train).reshape(-1, 1)
Y_test = np.array(Y_test).reshape(-1, 1)

if MODEL:
    # Creating the RNN model
    model = Sequential()
    model.add(LSTM(units=64, input_shape=(k, 4), return_sequences=True))
    model.add(LSTM(units=128, return_sequences=True))
    model.add(LSTM(units=256))
    model.add(Dropout(0.2))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))
    model.summary()
else:
    # Creating the SAN model
    model = build_san(input_shape=(k, 4), num_head=4, d_model=64, ff_dim=64, dropout=0.2)
    model.summary()


# Optimizer
optimizer = Adam(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
    )

# compile the model
model.compile(
    optimizer=optimizer,
    loss="mean_squared_error",
    metrics=[]
    )

# saving best model while training
cp_path = "./trained/"+folder+"/model"
checkpoint = ModelCheckpoint(
    cp_path,
    monitor="val_loss",
    save_best_only=True,
    save_weights_only=False,
    verbose=1,
    mode="auto"
    )


def scheduler(epoch):
    """adaptable learning rate"""
    if epoch < 20:  # Gradually decrease for the first 20 epochs
        return 1e-2 / (10 * (epoch/2+1))
    else:
        return 1e-4


lr_scheduler = LearningRateScheduler(scheduler)
epochs = 20

if RUN_MODEL:
    # Train the RNN model
    h = model.fit(
        X_train,
        Y_train,
        epochs=epochs,
        batch_size=32,
        validation_data=[X_test, Y_test],
        callbacks=[checkpoint, lr_scheduler]
        )

    with open('./trained/'+folder+'/history.txt', 'w') as f:
        f.write(str(h.history))
        f.close()

if EVAL_MODEL:
    # load best model saved
    model = load_model(
        cp_path,
        custom_objects=None,
        compile=True
        )

    with open('./trained/'+folder+'/history.txt', 'r') as f:
        history = ast.literal_eval(f.read())

    # Plot loss across epochs and interpret it
    plt.figure()
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.xlim(-0.5, epochs - 0.5)
    plt.xticks(range(0, epochs), np.linspace(1, epochs, epochs, dtype=int))
    plt.ylabel('Loss')
    plt.legend(['train', 'test'], loc='upper right')
    # plt.show()
    plt.savefig('./trained/'+folder+'/loss_plot.png')

    # Plot learning rate
    plt.figure()
    plt.plot(history['lr'])
    plt.title('Model lr')
    plt.xlabel('Epoch')
    plt.xlim(-0.5, epochs - 0.5)
    plt.xticks(range(0, epochs), np.linspace(1, epochs, epochs, dtype=int))
    plt.ylabel('Learning rate')
    # plt.show()
    plt.savefig('./trained/'+folder+'/lr_plot.png')

    # Evaluate the model using MSE and MAE
    mse = model.evaluate(X_test, Y_test)

    with open('trained/'+folder+'/evaluation.txt', 'w') as f:
        f.write(f'Mean Squared Error (MSE): {mse:.4f}\n')
        f.close()

    # predict cbg
    pred_value = model.predict(X_test)

    # calc diff
    diff = Y_test - pred_value

    # Plot comparison between prediction and ground-truth
    plt.figure()
    plt.plot(pred_value)
    plt.plot(Y_test)
    plt.title(f'Comparison of prediction and ground-truth\n ({folder})')
    plt.xlim(500, 1000)
    plt.xlabel('Timestamps in 5 Minute Intervals')
    plt.ylabel('cbg normalized')
    plt.legend(['predicted', 'truth'], loc='upper right')
    # plt.show()
    plt.savefig('./trained/' + folder + '/compare_plot.png')

    # Plot (absolute) differences between prediction and ground-truth
    plt.figure()
    plt.plot(diff, label=f'Differences')
    plt.plot(abs(diff), label='Absolute Differences')
    plt.axhline(np.sum(abs(diff)) / len(diff), color='r', linestyle='--', label='Mean absolute difference')
    plt.title(f'Difference prediction and ground-truth\n ({folder})')
    plt.xlim(1000, 2000)
    plt.xlabel('Timestamps in 5 Minute Intervals')
    plt.ylabel('predicted cbg normalized')
    plt.legend(loc='upper right')
    # plt.show()
    plt.savefig('./trained/' + folder + '/difference_plot.png')

    # reverse normalization
    backtrs_pred_value = pred_value * param_max['cbg']
    backtrs_Y_test = Y_test * param_max['cbg']
    backtrs_diff = backtrs_Y_test - backtrs_pred_value

    # Plot comparison between prediction and ground-truth (backtransformed)
    plt.figure()
    plt.plot(backtrs_pred_value)
    plt.plot(backtrs_Y_test)
    plt.title(f'Comparison of prediction and ground-truth\n ({folder})')
    plt.xlim(500, 1000)
    plt.xlabel('Timestamps in 5 Minute Intervals')
    plt.ylabel('cbg / mg/dL')
    plt.legend(['predicted', 'truth'], loc='upper right')
    # plt.show()
    plt.savefig('./trained/' + folder + '/compare_plot_backtransformed.png')

    # Plot (absolute) differences between prediction and ground-truth (backtransformed)
    plt.figure()
    plt.plot(backtrs_diff, label='Differences')
    plt.plot(abs(backtrs_diff), label='Absolute Differences')
    plt.axhline(np.sum(abs(backtrs_diff)) / len(backtrs_diff), color='r', linestyle='--', label='Mean absolute difference')
    plt.title(f'Difference prediction and ground-truth\n ({folder})')
    plt.xlim(1000, 2000)
    plt.xlabel('Timestamps in 5 Minute Intervals')
    plt.ylabel('cbg difference / mg/dL')
    plt.legend(loc='upper right')
    # plt.show()
    plt.savefig('./trained/' + folder + '/difference_plot_backtransformed.png')