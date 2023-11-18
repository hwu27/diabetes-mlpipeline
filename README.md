# diabetes-mlpipeline

This ML Pipeline is for the Pima Indians Diabetes Database

The dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases

The objective is to predict whether or not a patient has diabetes, based on certain diagnostic measurements included in the dataset

The dataset consists of several medical predictor variables and one target variable, Outcome

Predictor variables includes the number of pregnancies the patient has had, their BMI, insulin level, age, and so on

The target variable is whether or not the patient has diabetes

The dataset is available at https://www.kaggle.com/uciml/pima-indians-diabetes-database

The dataset is also available at https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv

Pipeline was inspired by https://www.kaggle.com/code/pouryaayria/a-complete-ml-pipeline-tutorial-acu-86#6.-Ensemble-Methods

Features many ML algorithms and ensemble methods for classification 

Features hyperparameter tuning using RandomizedSearchCV and GridSearchCV 

Features stacking ensemble method

Features error correlation between models

Features imputation of missing values using IterativeImputer

Features outlier detection using IQR method

Features scaling of features using StandardScaler

Features cross-validation techniques such as StratifiedKFold and KFold

Features evaluation metrics such as accuracy, precision, recall, f1, and roc_auc

Features plotting of accuracies of each model

Features plotting of error correlation between models

Detailed notes are provided for each step of the ML Pipeline for learning purposes
Final result of Roc_Auc: 0.808 (+/- 0.067) for the ensemble of models
