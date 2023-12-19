"""
Complete Machine Learning Pipeline for the Pima Indians Diabetes Dataset
"""

import pandas as pd
import numpy as np
import random
from matplotlib import pyplot 
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from imblearn.over_sampling import SMOTE

import plotly.graph_objs as go
import seaborn as sns
from matplotlib import pyplot 



SEED = 27
random.seed(SEED)

# Load the dataset
df = pd.read_csv('archive/diabetes.csv')
"""
# Basics of Data Exploration
print(df.shape)
print(df.info())
print(df.head())
"""
# Check for outliers, missing values, and distributions
"""
pyplot.rcParams['figure.figsize'] = [20, 10]
df.hist()
pyplot.show()
"""
# Replace zeros in specific columns with NaN
columns_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[columns_with_zeros] = df[columns_with_zeros].replace(0, np.nan)

# Apply MICE imputation
mice_imputer = IterativeImputer()
df_imputed = mice_imputer.fit_transform(df)
# Convert the numpy array returned by MICE back to a pandas DataFrame
df_imputed = pd.DataFrame(df_imputed, columns=df.columns)

# IQR method to detect outliers
Q1 = df_imputed.quantile(0.25)
Q3 = df_imputed.quantile(0.75)
IQR = Q3 - Q1
# Define a mask to identify rows with outliers
outlier_mask = ((df_imputed < (Q1 - 1.5 * IQR)) | (df_imputed > (Q3 + 1.5 * IQR))).any(axis=1)
# Filter out the outliers
filtered_df = df_imputed[~outlier_mask]
# Getting dataframe columns names
filtered_df_name=filtered_df.columns

# Split dataset into training set and test set
X = filtered_df[filtered_df_name[0:8]] # Features
Y = filtered_df[filtered_df_name[8]] # Target variable
X_train, X_test, y_train, y_test= train_test_split(X, Y, test_size=0.2, random_state=SEED, stratify=Y)


"""
# Check the new distribution of the dataset
pyplot.title('Distribution of the dataset after imputation, IQR, and SMOTE')
pyplot.rcParams['figure.figsize'] = [20, 10]
filtered_df.hist()
pyplot.show()
"""
# Apply SMOTE to the training data to balance the dataset (oversampling)
pipeline = SMOTE(random_state=SEED)
X_train, y_train = pipeline.fit_resample(X_train, y_train)
# Lets check the number of outcomes in each class
# print(pd.Series(y_train).value_counts())

# Inspired by https://www.kaggle.com/code/pouryaayria/a-complete-ml-pipeline-tutorial-acu-86#Define-Problem:
# Minmax seems to do better than standard scaler
minmax_scaler = MinMaxScaler()
X_train = minmax_scaler.fit_transform(X_train)
X_test = minmax_scaler.transform(X_test)


# Function to get a list of models to evaluate
def GetModel():
    basedModels = []
    basedModels.append(('NB', GaussianNB())),
    basedModels.append(('LR', LogisticRegression(random_state=SEED))),
    basedModels.append(('KNN', KNeighborsClassifier())),
    basedModels.append(('SVM', SVC(probability=True, random_state=SEED))),
    basedModels.append(('CART', DecisionTreeClassifier(random_state=SEED)))

    basedModels.append(('ada', AdaBoostClassifier(random_state=SEED))),
    basedModels.append(('gb', GradientBoostingClassifier(random_state=SEED))),
    basedModels.append(('xgb', XGBClassifier(random_state=SEED))),
    basedModels.append(('lgb', LGBMClassifier(random_state=SEED))),
    basedModels.append(('cat', CatBoostClassifier(verbose=False, random_state=SEED)))

    basedModels.append(('rf', RandomForestClassifier(random_state=SEED)))
    return basedModels

# Dictionary of possible parameters to evaluate
param_dict = {
    'NB': {},
    'LR': { 
        'C': np.logspace(-4, 4, 50),  
        'penalty': ['l2', None] 
    },
    'KNN': {
        'n_neighbors': range(1, 51),  
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski', 'chebyshev']
    },
    'SVM': {
        'C': np.logspace(-4, 4, 10), 
        'gamma': np.logspace(-4, 4, 10),  
        'kernel': ['rbf', 'sigmoid', 'linear']
    },
    'CART': {
        'max_depth': list(range(1, 21)), 
        'min_samples_split': range(2, 31, 2), 
        'min_samples_leaf': range(1, 31, 2)  
    },
    'ada': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 1.0],
        'algorithm': ['SAMME', 'SAMME.R']
    },
    'gb': {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 4, 6],
        'min_samples_leaf': [1, 2, 4],
        'subsample': [0.8, 0.9, 1.0],
        'max_features': ['sqrt', 'log2', None]
    },
    'xgb': {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'min_child_weight': [1, 3, 5],
        'gamma': [0, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'reg_alpha': [0, 0.1, 0.2],
        'reg_lambda': [0, 0.1, 0.2]
    },
    'lgb': {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'num_leaves': [31, 50, 70],
        'max_depth': [3, 5, 7],
        'min_child_samples': [20, 30, 50],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    },
    'cat': {
        'iterations': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'depth': [4, 6, 8],
        'l2_leaf_reg': [1, 3, 5],
        'border_count': [32, 64, 128],
        'bagging_temperature': [0.0, 1.0, 2.0]
    },
    'rf': {
        'n_estimators': np.arange(100, 1001, 100),
        'max_depth': np.arange(10, 31, 5),
        'min_samples_split': range(2, 11),
        'min_samples_leaf': range(1, 11),
        'max_features': ['sqrt', 'log2'],
        'criterion': ['gini', 'entropy']
    }
}

# Define cross-validation strategy
stratified_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)

# Function to apply RandomizedSearchCV to each model
def ApplyRandomizedSearch(models, param_dist, X_train, y_train, n_iter=100, cv=stratified_cv, random_state=SEED):
    """
    Apply RandomizedSearchCV to a list of models with specified parameter distributions.
    
    Parameters:
    - models: List of (name, model) tuples
    - param_dist: Dictionary of parameter distributions {model_name: distribution}
    - X_train, y_train: Training data and labels
    - n_iter: Number of parameter settings sampled (default=100)
    - cv: Cross-validation strategy (default=stratified_cv)
    - random_state: Random state for reproducibility (default=SEED)
    
    Returns:
    - best_params: Dictionary of best parameters for each model
    - best_scores: Dictionary of best scores for each model
    """
    best_params = {}
    best_scores = {}

    for name, model in models:
        if name in param_dist:
            print(f"Optimizing {name}...")
            rsearch = RandomizedSearchCV(estimator=model, param_distributions=param_dist[name], n_iter=n_iter, cv=cv, verbose=2, random_state=random_state, n_jobs=-1)
            rsearch.fit(X_train, y_train)
            best_params[name] = rsearch.best_params_
            best_scores[name] = rsearch.best_score_
            print(f"Best Params for {name}: {rsearch.best_params_}")
            print(f"Best Score for {name}: {rsearch.best_score_}\n")
    return best_params, best_scores

"""
best_params, best_scores = ApplyRandomizedSearch(GetModel(), param_dict, X_train, y_train, n_iter=100, cv=stratified_cv, random_state=SEED)
print(best_params)
print(best_scores)
"""

# Dictionary of best parameters for each model
best_param_models = {
    'NB': GaussianNB(),
    'LR': LogisticRegression(penalty='l2', C=7.9060432109076855, random_state=SEED),
    'KNN': KNeighborsClassifier(weights='distance', n_neighbors=2, metric='euclidean'),
    'SVM': SVC(kernel='rbf', gamma=21.54434690031882, C=1291.5496650148827, probability=True, random_state=SEED),
    'CART': DecisionTreeClassifier(min_samples_split=18, min_samples_leaf=15, max_depth=6, random_state=SEED),
    'ada': AdaBoostClassifier(n_estimators=200, learning_rate=0.1, algorithm='SAMME.R', random_state=SEED),
    'gb': GradientBoostingClassifier(subsample=0.9, n_estimators=200, min_samples_split=2, 
                                     min_samples_leaf=1, max_features=None, max_depth=7, 
                                     learning_rate=0.1, random_state=SEED),
    'xgb': XGBClassifier(subsample=0.8, reg_lambda=0.2, reg_alpha=0.2, n_estimators=100, 
                         min_child_weight=1, max_depth=5, learning_rate=0.2, gamma=0.1, 
                         colsample_bytree=0.8, random_state=SEED),
    'lgb': LGBMClassifier(subsample=0.8, num_leaves=70, n_estimators=200, 
                          min_child_samples=20, max_depth=5, learning_rate=0.2, 
                          colsample_bytree=0.8, verbose=-1, random_state=SEED),
    'cat': CatBoostClassifier(learning_rate=0.2, l2_leaf_reg=1, iterations=200, depth=8, 
                              border_count=64, bagging_temperature=1.0, verbose=False, random_state=SEED),
    'rf': RandomForestClassifier(n_estimators=500, min_samples_split=3, min_samples_leaf=1, max_features='sqrt', max_depth=15, criterion='gini', random_state=SEED)
}

# Scoring metrics for evaluation
scoring_metrics = ['accuracy', 'roc_auc', 'f1', 'precision', 'recall']

# Function to evaluate each model with cross-validation
def CVScoreModel(models, X, y, scoring_metrics=scoring_metrics, cv=stratified_cv):
    results = []
    for model_name, model in models.items():
        cv_results = cross_validate(model, X, y, scoring=scoring_metrics, cv=cv, n_jobs=-1)
        results.append({
            'Model': model_name,
            **{metric: np.mean(cv_results[f'test_{metric}']) for metric in scoring_metrics}
        })
    df_results = pd.DataFrame(results)
    return df_results
"""
scores = CVScoreModel(best_param_models, X_train, y_train, scoring_metrics=scoring_metrics, cv=stratified_cv)
print(scores)
"""
"""
# Create a voting classifier
estimator_model = [('SVM', best_param_models['SVM']), ('LR', best_param_models['LR']), ('ada', best_param_models['ada'])]
voting = VotingClassifier(estimators=estimator_model, voting='soft', weights=[1.6,1.173, 1.6])

# test on cross validation
scores = cross_validate(voting, X_train, y_train, scoring=scoring_metrics, cv=stratified_cv, n_jobs=-1)

# Calculating averages for each metric
average_scores = {metric: np.mean(scores[f'test_{metric}']) for metric in scoring_metrics}

# Print average scores
for metric, avg_score in average_scores.items():
    print(f"Average {metric}: {avg_score:.4f}")
"""

# Create a voting classifier
estimator_model = [('cat', best_param_models['cat']), ('svm', best_param_models['SVM']), ('ada', best_param_models['ada']), ('lr', best_param_models['LR'])]
voting = VotingClassifier(estimators=estimator_model, voting='soft', weights=[1.6, 1.71, 2.5, 1])

# test on cross validation
scores = cross_validate(voting, X_train, y_train, scoring=scoring_metrics, cv=stratified_cv, n_jobs=-1)

# Calculating averages for each metric
average_scores = {metric: np.mean(scores[f'test_{metric}']) for metric in scoring_metrics}

# Print average scores
for metric, avg_score in average_scores.items():
    print(f"Average {metric}: {avg_score:.4f}")



"""
# Create a voting classifier for all models
estimator_model = [('NB', best_param_models['NB']), ('LR', best_param_models['LR']), ('KNN', best_param_models['KNN']), ('SVM', best_param_models['SVM']), ('CART', best_param_models['CART']), ('ada', best_param_models['ada']), ('gb', best_param_models['gb']), ('xgb', best_param_models['xgb']), ('lgb', best_param_models['lgb']), ('cat', best_param_models['cat']), ('rf', best_param_models['rf'])]
voting = VotingClassifier(estimators=estimator_model, voting='soft')

# test on cross validation
scores = cross_validate(voting, X_train, y_train, scoring=scoring_metrics, cv=stratified_cv, n_jobs=-1)

# Calculating averages for each metric
average_scores = {metric: np.mean(scores[f'test_{metric}']) for metric in scoring_metrics}

# Print average scores
for metric, avg_score in average_scores.items():
    print(f"Average {metric}: {avg_score:.4f}")


# Train each model and collect their predictions
predictions = []
for name, model in best_param_models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)  # Assume X_val is your validation set
    predictions.append(preds)

# Calculate correlation matrix
# Convert predictions to a boolean format (correct or incorrect)
correct_predictions = [preds == y_test for preds in predictions]
correlation_matrix = np.corrcoef(correct_predictions)

# Visualize with heatmap
sns.heatmap(correlation_matrix, annot=True, xticklabels=[name for name, _ in best_param_models.items()], yticklabels=[name for name, _ in best_param_models.items()])
pyplot.title('Model Prediction Correlation')
pyplot.show()
"""

