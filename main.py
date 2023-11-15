# -------------------- Imports -----------------------------------------------------------------------------------------
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.optimizers import Adam

# -------------------- Coding ------------------------------------------------------------------------------------------

visu = False  # Visualization of Data ON/OFF

# Load Data
path = "Ohio Data"

train_data = []
test_data = []

for year in os.listdir(path):
    train_path = os.path.join(path, year, "train")
    test_path = os.path.join(path, year, "test")

    for root, dirs, files in os.walk(train_path):
        for file in files:
            if file.endswith('.csv'):  # Check if the file is a CSV file
                file_path = os.path.join(root, file)  # full file path
                data = pd.read_csv(file_path)
                train_data.append(data)

    for root, dirs, files in os.walk(test_path):
        for file in files:
            if file.endswith('.csv'):  # Check if the file is a CSV file
                file_path = os.path.join(root, file)  # full file path
                data = pd.read_csv(file_path)
                test_data.append(data)

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

if visu:
    for i in range(len(test_data)):
        plt.subplot(3, 4, i+1)
        plt.plot(test_data[i]['cbg'])
        plt.xlabel('Timestamps in 5 Minute Intervals')
        plt.ylabel('CBG')
        plt.title('CBG over Time'+str(i))
    plt.show()

# TODO: Parameter changes?
k = 100  # Number of past BG values
PH = 5  # Prediction horizon


# Prepare sequences for the RNN model
def sequences(patient_data, window, horizon):
    X = []
    Y = []
    for patient in range(len(patient_data)):
        for idx in range(len(patient_data[patient]) - window - horizon + 1):
            if max(patient_data[patient]['missing_cbg'].values[idx:idx + window + horizon]) == 0:
                X.append(patient_data[patient]['cbg'].values[idx:idx + window])
                Y.append(patient_data[patient]['cbg'].values[idx + window + horizon - 1])
    return [np.array(X), np.array(Y)]

# TODO: Data normalizing between 0 & 1 ?
# TODO: scipy.cubic_spline input missing data in cbg
# TODO: fill up the carbs and insulin with 0es => add into input (correlation matrix which features are important)

X_train, Y_train = sequences(train_data, k, PH)
X_test, Y_test = sequences(test_data, k, PH)

# TODO: 2 experiments:
#  patient separately => 12 models
#  all together generalized (without test patient) => 12 models

# Reshaping
X_train = X_train.reshape(-1, k, 1)
X_test = X_test.reshape(-1, k, 1)
Y_train = Y_train.reshape(-1, 1)
Y_test = Y_test.reshape(-1, 1)

# TODO: possible model changes: activation function because of normalized data
# TODO: (k, 4) if four features cbg, insulin basal, bolus, carb
# Creating the RNN model
rnn_model = Sequential()
rnn_model.add(LSTM(units=64, input_shape=(k, 1), return_sequences=True))
rnn_model.add(LSTM(units=64))
rnn_model.add(Dropout(0.2))
rnn_model.add(Dense(units=1, activation='linear'))
rnn_model.summary()

# Optimizer
optimizer = Adam(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
    )

# compile the model
rnn_model.compile(
    optimizer=optimizer,
    loss="mean_squared_error",
    metrics=[]
    )

# saving best model while training
cp_path = "./trained/model"
checkpoint = ModelCheckpoint(
    cp_path,
    monitor="val_loss",
    save_best_only=True,
    save_weights_only=False,
    verbose=1,
    mode="auto"
    )


# adaptable learning rate
def scheduler(epoch):
    if epoch < 20:  # Gradually decrease for the first 20 epochs
        return 1e-2 / (10 * (epoch/2+1))
    else:
        return 1e-4


lr_scheduler = LearningRateScheduler(scheduler)
epochs = 20

# Train the RNN model
h = rnn_model.fit(
    X_train,
    Y_train,
    epochs=epochs,
    batch_size=32,
    validation_data=[X_test, Y_test],
    callbacks=[checkpoint, lr_scheduler]
    )

with open('./trained/history.txt', 'w') as f:
    f.write(str(h.history))
    f.close()

# Plot loss across epochs and interpret it
plt.figure()
plt.plot(h.history['loss'])
plt.plot(h.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.xlim(-0.5, epochs-0.5)
plt.xticks(range(0, epochs), np.linspace(1, epochs, epochs, dtype=int))
plt.ylabel('Loss')
plt.legend(['train', 'test'], loc='upper right')
# plt.show()
plt.savefig('./trained/loss_plot.png')

# load best model saved
rnn_model = load_model(
    cp_path,
    custom_objects=None,
    compile=True
    )

# Evaluate the model using MSE and MAE
mse = rnn_model.evaluate(X_test, Y_test)

with open('trained/evaluation.txt', 'w') as f:
    f.write(f'Mean Squared Error (MSE): {mse:.4f}\n')
    f.close()

# TODO: visualization of predicted values compared to true values to check the validity
