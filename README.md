# Blood-Glucose-Prediction
Hello world!
## How to run our experiments?
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

We hereby confirm, that no form of AI has been used for the creation, writing and correction of either the code or of the report.




