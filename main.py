# -------------------- Imports -----------------------------------------------------------------------------------------
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------- Coding ------------------------------------------------------------------------------------------

# Load Data
path = "Ohio Data"

dataframe = []

for root, dirs, files in os.walk(path):
    for file in files:
        if file.endswith('.csv'):  # Check if the file is a CSV file
            file_path = os.path.join(root, file)  # Obtain the full file path
            # Read the CSV file and append its contents to the combined_data DataFrame
            data = pd.read_csv(file_path)
            dataframe.append(data)

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


for i in range(len(dataframe)):
    plt.subplot(4,6,i+1)
    plt.plot(dataframe[i]['cbg'])
    plt.xlabel('Timestamps in 5 Minute Intervals')
    plt.ylabel('CBG')
    plt.title('CBG over Time'+str(i))
plt.show()
