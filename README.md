# Blood-Glucose  Prediction

Hello world! <br>
This project is based on the prediction of blood glucose for patients with type 1 diabetes using deep learning. A Recurrent Neural Network (RNN) and a Self-Acting Neural network (SAN) models were used. The OhioT1DM dataset containing data of 12 patients was used for training and testing the models. 



## Pipeline structure

**Global parameters**: Definition of global parameters to control visualization, model training and evaluation. <br>

**Loading data**: Loading data from the specified location in CSV format, organized by year, divided into training and test sets.

**Data preparation**: _prepareData()_ interpolates missing data and normalizes the specified columns. Optional visualization shows graphs of original and interpolated data for each patient.

**Correlation matrix**: _correlationMatrix()_ calculates the correlation matrix between variables, highlighting the correlation with the expected glucose level in the future.

**Neural model construction**:
* SAN: _build_san()_ defines a neural network model with self-attention levels for glucose prediction.
* RNN: Built with three LSTM layers (64, 128, 256 units) followed by 20% dropout, dense layers with ReLU activation and a sigmoid output. Receives input of k-long sequences with information on glucose, basal insulin, carbohydrate, and insulin bolus as specified in the report.

**Creation of sequences to train the model**: _sequences()_ creates input (X) and output (Y) sequences from patient data, using specified time windows and prediction horizons.

**Model training**: The code trains a neural network model (RNN or SAN) using the [Adam optimizer](https://www.google.com/search?client=firefox-b-d&q=adam+optimization), the loss "mean_squared_error" and a learning rate reduction function.

The scheduler function implements a learning rate adjustment during model training. Specifically, for the first 20 epochs, the learning rate gradually decreases, while for subsequent epochs it remains constant at 1e-4.

**Saving Model**: Saves model in a directory specific to the current experiment.

**Model Evaluation**: If enabled, the code loads the saved model and evaluates performance, generating graphs for loss, learning rate, and comparing predicted results with actual results.

**Saving Results**: Saves results and graphs in a directory specific to the current experiment.



## Global Parameters: How to run our experiments?

1. VISU_PLOTS = True / False
    1. Choose whether plots of raw and prepared data should be created and saved.
2. VISU_CORR = True / False
    1. Choose whether the (boxplot of) correlation matrices should be created, displayed, and saved.
3. RUN_MODEL = True / False
    1. Choose whether the corresponding model should be trained or not.
4. EVAL_MODEL = True / False
    1. Choose whether the corresponding model should be evaluated or not.
    2. This can be used to load a previously created model.
5. MODEL = True / False
    1. Choose what model you want to train True == RNN / False == SAN.
6. MODE = True / False
    1. Choose which mode should be trained: True == Separately / False == Generalized.
7. PATIENT_TBD = (int)
    1. Insert the unique patient number which should be trained and tested (separately) or exclusively tested (generalized)
8. Window length "k" = (int)
    1. Insert the desired length of the segments which will be created for the training of the model. (Step size is equal to 5 min intervals)
9. Prediction horizon "PH" = (int)
    1. Insert the temporal difference between the current and predicted cbg value (Step size is equal to 5 min intervals)



## Contributions


| Legend  | Contribution | 
| ------------- |:-------------:|
| Matteo Tagliabue      | Utilities, Report, AI training (UBELIX)    |
| Patrick von Raumer      | AI-Setup, Report, AI training (UBELIX)     |
| Yves Moerlen      | AI-Setup, Report, AI training (UBELIX)   |
| Matthias Pracht      | Utilities, Report, AI training (UBELIX)   |



### Disclaimer

We hereby confirm, that no form of AI has been used for the creation, writing and correction of either the code or of the report, with the exception of partial translations using DeepL.




