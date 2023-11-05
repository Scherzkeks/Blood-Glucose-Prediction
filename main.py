# -------------------- Imports -----------------------------------------------------------------------------------------
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

k = 100  # Number of past BG values
PH = 5  # Prediction horizon


# Prepare sequences for the RNN model
def sequences(patient, k, PH):
    X = []
    Y = []
    for j in range(len(patient)):
        for i in range(len(patient[j]) - k - PH + 1):
            if max(patient[j]['missing_cbg'].values[i:i+k]) == 0:
                X.append(patient[j]['cbg'].values[i:i+k])  # Input: 'k' BG values
                Y.append(patient[j]['cbg'].values[i+k + PH - 1])
    return [np.array(X), np.array(Y)]


X_train, Y_train = sequences(train_data, k, PH)
X_test, Y_test = sequences(test_data, k, PH)

# Creating the RNN model
rnn_model = Sequential()
rnn_model.add(LSTM(50, return_sequences=True, input_shape=(k, 1)))
rnn_model.add(LSTM(50, return_sequences=True))
rnn_model.add(LSTM(50))
rnn_model.add(Dropout(0.2))
rnn_model.add(Dense(1, activation='linear'))

# Optimizer
optimizer = Adam(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
    )

# compile the model
rnn_model.compile(
    optimizer=optimizer,
    loss="bce",
    metrics=["accuracy", keras.metrics.Precision(), keras.metrics.Recall()]
    )

cp_path = "./trained/model"
checkpoint = ModelCheckpoint(
    cp_path,
    monitor="val_accuracy",
    save_best_only=True,
    save_weights_only=False,
    verbose=1,
    mode="auto"
    )


# adaptable learning rate
def scheduler(epoch):
    if epoch < 1:
        return 1e-5
    else:
        return 1e-6


lr_scheduler = LearningRateScheduler(scheduler)
epochs = 2

# Train the RNN model
h = rnn_model.fit(
    X_train,
    Y_train,
    epochs=epochs,
    batch_size=16,
    validation_data=[X_test, Y_test],
    callbacks=[checkpoint, lr_scheduler]
    )

with open('./trained/history.txt', 'w') as f:
    f.write(str(h.history))
    f.close()

# Plot Accuracy across epochs and interpret it
plt.figure()
plt.plot(h.history['accuracy'])
plt.plot(h.history['val_accuracy'])
plt.title('Model from scratch Accuracy')
plt.xlabel('Epoch')
plt.xlim(-0.5, epochs-0.5)
plt.xticks(range(0, epochs), np.linspace(1, epochs, epochs, dtype=int))
plt.ylabel('Accuracy')
plt.legend(['train', 'test'], loc='upper left')
# plt.show()
plt.savefig('./trained/sc_accuracy_plot.png')

# Plot loss across epochs and interpret it
plt.figure()
plt.plot(h.history['loss'])
plt.plot(h.history['val_loss'])
plt.title('Model from scratch Loss')
plt.xlabel('Epoch')
plt.xlim(-0.5, epochs-0.5)
plt.xticks(range(0, epochs), np.linspace(1, epochs, epochs, dtype=int))
plt.ylabel('Loss')
plt.legend(['train', 'test'], loc='upper right')
# plt.show()
plt.savefig('./trained/sc_loss_plot.png')

# load best model saved
rnn_model = load_model(
    cp_path,
    custom_objects=None,
    compile=True
    )

# compute the model accuracy, loss, precision and recall using model.evaluate and print them
loss, accuracy, precision, recall = rnn_model.evaluate(X_test, Y_test)

# Print the evaluation metrics and precision/recall
with open('trained/evaluation.txt', 'w') as f:
    f.write(f'Loss: {100*loss:.2f}\n')
    f.write(f'Accuracy: {100*accuracy:.2f}\n')
    f.write(f'Precision: {100*precision:.2f}\n')
    f.write(f'Recall: {100*recall:.2f}\n')
    f.close()

