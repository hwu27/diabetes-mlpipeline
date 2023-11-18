# This ML Pipeline is for the Pima Indians Diabetes Database
# The dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases
# The objective is to predict whether or not a patient has diabetes, based on certain diagnostic measurements included in the dataset
# The dataset consists of several medical predictor variables and one target variable, Outcome
# Predictor variables includes the number of pregnancies the patient has had, their BMI, insulin level, age, and so on
# The target variable is whether or not the patient has diabetes
# The dataset is available at https://www.kaggle.com/uciml/pima-indians-diabetes-database
# The dataset is also available at https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv
# Pipeline was inspired by https://www.kaggle.com/code/pouryaayria/a-complete-ml-pipeline-tutorial-acu-86#6.-Ensemble-Methods
# Features many ML algorithms and ensemble methods for classification 
# Features hyperparameter tuning using RandomizedSearchCV and GridSearchCV 
# Features stacking ensemble method
# Features error correlation between models
# Features imputation of missing values using IterativeImputer
# Features outlier detection using IQR method
# Features scaling of features using StandardScaler
# Features cross-validation techniques such as StratifiedKFold and KFold
# Features evaluation metrics such as accuracy, precision, recall, f1, and roc_auc
# Features plotting of accuracies of each model
# Features plotting of error correlation between models
# Detailed notes are provided for each step of the ML Pipeline for learning purposes
# Final result of Roc_Auc: 0.808 (+/- 0.067) for the ensemble of models

# Load librariess
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import StackingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate
from xgboost import XGBClassifier

# Set a seed for reproducibility
SEED = 7
np.random.seed(SEED) 

# Load dataset  
df = pd.read_csv('archive/diabetes.csv')

# Replace zeros with NaN in specific columns
columns_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[columns_with_zeros] = df[columns_with_zeros].replace(0, np.nan)

# Create the imputer
iterative_imputer = IterativeImputer()

# Apply the imputer
df[columns_with_zeros] = iterative_imputer.fit_transform(df[columns_with_zeros])

#--------------------------------------------------------------------------------------
""" For if we chose to replace the zeros with the median of each column
columns_with_nans = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

# Replace NaN values with the median of each column
for column in columns_with_nans:
    median_value = df[column].median()
    df[column].fillna(median_value, inplace=True)

"""
#IQR method to detect outliers
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

# Define a mask to identify rows with outliers
outlier_mask = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)

# Filter out the outliers
filtered_df = df[~outlier_mask]

# Getting dataframe columns names
filtered_df_name=filtered_df.columns

# Split-out validation dataset
X = filtered_df[filtered_df_name[0:8]] # Features
Y = filtered_df[filtered_df_name[8]] # Target variable
X_train, X_test, y_train, y_test= train_test_split(X, Y, test_size=0.3, random_state=0, stratify=Y) 
# 70% training and 30% test: Stratify preserves the proportion of the target variable's classes as in the original dataset

#--------------------------------------------------------------------------------------

# Define a function to create a baseline set of models
def GetBasedModel():
    basedModels = []
    basedModels.append(('LR', Pipeline([('Scaler', StandardScaler()), ('LR', LogisticRegression())])))
    basedModels.append(('LDA', Pipeline([('Scaler', StandardScaler()), ('LDA', LinearDiscriminantAnalysis())])))
    basedModels.append(('KNN', Pipeline([('Scaler', StandardScaler()), ('KNN', KNeighborsClassifier())])))
    basedModels.append(('CART', Pipeline([('Scaler', StandardScaler()), ('CART', DecisionTreeClassifier())])))
    basedModels.append(('NB', Pipeline([('Scaler', StandardScaler()), ('NB', GaussianNB())])))
    basedModels.append(('SVM', Pipeline([('Scaler', StandardScaler()), ('SVM', SVC(probability=True))])))
    basedModels.append(('AB', Pipeline([('Scaler', StandardScaler()), ('AB', AdaBoostClassifier())])))
    basedModels.append(('GBM', Pipeline([('Scaler', StandardScaler()), ('GBM', GradientBoostingClassifier())])))
    basedModels.append(('RF', Pipeline([('Scaler', StandardScaler()), ('RF', RandomForestClassifier())])))
    basedModels.append(('ET', Pipeline([('Scaler', StandardScaler()), ('ET', ExtraTreesClassifier())])))
    return basedModels

# Define a function to evaluate each model in turn
def BasedLine2(X_train, y_train, models, seed=7):
    num_folds = 10
    scoring = 'accuracy'
    results = []
    names = []
    for name, model in models:
        try:
            # A cross-validation technique that ensures each fold of the dataset has the same proportion of observations with a given label
            kfold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed) 
            # A utility function that evaluates a score by cross-validation. 
            # It takes the model, training data (X_train, y_train), the cross-validation strategy (kfold), and the scoring metric (scoring, which is 'accuracy' in your case). 
            # n_jobs=-1 tells the function to use all available CPU cores for parallel computation.
            cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring, n_jobs=-1)
            results.append(cv_results)
            names.append(name)
            msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
            print(msg)
        except Exception as e:
            print(f"Model {name} failed to train with error: {e}")
    return names, results

# A dictionary of hyperparameters for each model
param_dist_dict = {
    'LR': { 
        'LR__C': np.logspace(-4, 4, 50),  
        'LR__penalty': ['l1', 'l2', 'elasticnet', 'none']  
    },
    'LDA': {
        'LDA__shrinkage': np.linspace(0.0, 1.0, 50), 
        'LDA__solver': ['svd', 'lsqr', 'eigen']
    },
    'KNN': {
        'KNN__n_neighbors': range(1, 51),  
        'KNN__weights': ['uniform', 'distance'],
        'KNN__metric': ['euclidean', 'manhattan', 'minkowski', 'chebyshev'] 
    },
    'CART': {
        'CART__max_depth': list(range(1, 21)), 
        'CART__min_samples_split': range(2, 31, 2), 
        'CART__min_samples_leaf': range(1, 31, 2)  
    },
    'NB': {},  # GaussianNB doesn't have hyperparameters that are typically tuned
    'SVM': {
        'SVM__C': np.logspace(-4, 4, 10), 
        'SVM__gamma': np.logspace(-4, 1, 10),  
        'SVM__kernel': ['rbf', 'sigmoid', 'linear']  
    },
    'AB': {
        'AB__n_estimators': np.arange(50, 501, 50),  
        'AB__learning_rate': np.linspace(0.01, 1, 20)  
    },
    'GBM': {
        'GBM__n_estimators': np.arange(50, 501, 50), 
        'GBM__learning_rate': np.linspace(0.01, 0.5, 20), 
        'GBM__max_depth': range(3, 15)  
    },
    'RF': {
        'RF__n_estimators': np.arange(100, 1001, 100), 
        'RF__max_features': ['sqrt', 'log2'],
        'RF__max_depth': list(range(3, 21)), 
        'RF__min_samples_split': range(2, 31, 2), 
        'RF__min_samples_leaf': range(1, 31, 2) 
    },
    'ET': {
        'ET__n_estimators': np.arange(100, 1001, 100), 
        'ET__max_features': ['sqrt', 'log2'],
        'ET__max_depth': list(range(3, 21)), 
        'ET__min_samples_split': range(2, 31, 2),
        'ET__min_samples_leaf': range(1, 31, 2)  
    }
}

# Function to apply RandomizedSearchCV to each model
def ApplyRandomizedSearch(models, param_dist, X_train, y_train, n_iter=100, cv=5, random_state=SEED):
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
# Function to apply GridSearchCV to each model
def ApplyGridSearch(models, param_grid, X_train, y_train, cv=5, scoring='accuracy', verbose=2, random_state=SEED):
    best_params = {}
    best_scores = {}
    
    for name, model in models:
        if name in param_grid:
            print(f"Optimizing {name}...")
            grid_search = GridSearchCV(estimator=model, param_grid=param_grid[name], cv=cv, scoring=scoring, verbose=verbose)
            grid_search.fit(X_train, y_train)
            best_params[name] = grid_search.best_params_
            best_scores[name] = grid_search.best_score_
            print(f"Best Params for {name}: {grid_search.best_params_}")
            print(f"Best Score for {name}: {grid_search.best_score_}\n")
        max_key = None
        max_value = float('-inf')

        for key, value in best_scores.items():
            if value > max_value:
                max_key = key
                max_value = value
    return best_params, best_scores, max_key, max_value
"""

#--------------------------------------------------------------------------------------

# Define a function to evaluate an ensemble of models
def EvaluateEnsemble(X_train, y_train, models, seed=7, scoring_metrics=['accuracy']):
    num_folds = 10
    results = {}
    names = []

    if not isinstance(models, list):  # Ensure models is a list
        models = [models]

    for model in models:
        name = type(model).__name__
        try:
            kfold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
            cv_results = cross_validate(model, X_train, y_train, cv=kfold, scoring=scoring_metrics, n_jobs=-1, return_train_score=False)

            results[name] = cv_results
            names.append(name)
            print(f"Results for {name}:")
            for metric in scoring_metrics:
                print(f"  {metric}: {np.mean(cv_results[f'test_{metric}']):.3f} (+/- {np.std(cv_results[f'test_{metric}']):.3f})")
        except Exception as e:
            print(f"Model {name} failed to train with error: {e}")

    return names, results

# Define a function to plot the accuracies of each model
def plot_model_accuracies_with_hyperparams(best_scores, best_params):
    # Plotting accuracies
    plt.figure(figsize=(12, 6))
    models = list(best_scores.keys())
    scores = [best_scores[model] for model in models]
    bars = plt.bar(models, scores, color='lightblue')

    # Printing hyperparameters
    for model in models:
        print(f"Best Hyperparameters for {model}: {best_params[model]}")

    # Adding the accuracy value on top of each bar
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.01, round(yval, 3), ha='center', va='bottom')

    plt.xlabel('Models')
    plt.ylabel('Best CV Score')
    plt.title('Best Model Performance Comparison')
    plt.xticks(rotation=45)
    plt.ylim([min(scores) - 0.05, max(scores) + 0.05])  # Adjust y-axis limits
    plt.show()


#models = GetBasedModel()

#best_params, best_scores = ApplyRandomizedSearch(models, param_dist_dict, X_train, y_train)

#plot_model_accuracies_with_hyperparams(best_scores, best_params)

#--------------------------------------------------------------------------------------

# Instantiate models with the best hyperparameters for each model if we removed the zeros of each column of unplausible values
lr = LogisticRegression(C=0.00021209508879201905, solver='saga', max_iter=1000)
lda = LinearDiscriminantAnalysis(solver='eigen', shrinkage=0.061224489795918366)
knn = KNeighborsClassifier(weights='uniform', n_neighbors=33, metric='manhattan')
cart = DecisionTreeClassifier(min_samples_split=4, min_samples_leaf=23, max_depth=20)
nb = GaussianNB()
svm = SVC(kernel='sigmoid', gamma=0.016681005372000592, C=21.54434690031882, probability=True)
ab = AdaBoostClassifier(n_estimators=100, learning_rate=0.06210526315789474)
gbm = GradientBoostingClassifier(n_estimators=50, max_depth=4, learning_rate=0.13894736842105262)
rf = RandomForestClassifier(n_estimators=200, min_samples_split=6, min_samples_leaf=9, max_features='sqrt', max_depth=15)
et = ExtraTreesClassifier(n_estimators=900, min_samples_split=12, min_samples_leaf=9, max_features='log2', max_depth=12)

""" The best hyperparameters for each model if we replaced the zeros with the median of each column
lr = LogisticRegression(C=0.019306977288832496, solver='lbfgs', max_iter=1000)
lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=0.18367346938775508)
knn = KNeighborsClassifier(weights='distance', n_neighbors=18, metric='minkowski')
cart = DecisionTreeClassifier(min_samples_split=28, min_samples_leaf=17, max_depth=7)
nb = GaussianNB()
svm = SVC(kernel='sigmoid', gamma=0.004641588833612782, C=2.782559402207126)
ab = AdaBoostClassifier(n_estimators=100, learning_rate=0.06210526315789474)
gbm = GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.01)
rf = RandomForestClassifier(n_estimators=600, min_samples_split=12, min_samples_leaf=1, max_features='log2', max_depth=13)
et = ExtraTreesClassifier(n_estimators=300, min_samples_split=30, min_samples_leaf=1, max_features='log2', max_depth=15)
"""
# Define base learners for the ensemble
base_learners = [
    ('lr', lr),
    ('knn', knn),
    ('cart', cart),
    ('svm', svm),
    ('ab', ab),
    ('gbm', gbm),
    ('rf', rf)
]

models_dict = dict(base_learners)

for name, model in models_dict.items():
    model.fit(X_train, y_train)  # Fit each model

def calculate_error_correlation(models, X, y):
    """
    Calculate the error correlation between models.

    :param models: Dictionary of models
    :param X: Feature data
    :param y: True labels
    :return: DataFrame of error correlations
    """
    # Generate predictions for each model
    predictions = {name: model.predict(X) for name, model in models.items()}

    # Calculate error vectors
    error_vectors = {name: (pred != y) for name, pred in predictions.items()}

    # Initialize an empty DataFrame for error correlations
    error_corr = pd.DataFrame(index=models.keys(), columns=models.keys())

    # Calculate error correlation
    for model_a in error_vectors:
        for model_b in error_vectors:
            if model_a != model_b:
                corr = np.corrcoef(error_vectors[model_a], error_vectors[model_b])[0, 1]
                error_corr.loc[model_a, model_b] = corr
            else:
                error_corr.loc[model_a, model_b] = np.nan  # Avoid self-comparison

    return error_corr

# Usage example

#error_correlation_matrix = calculate_error_correlation(models_dict, X_train, y_train)
#print(error_correlation_matrix)

"""
# Define a list of meta-learners
meta_learners = {
    "Logistic Regression": LogisticRegression(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "Random Forest": RandomForestClassifier(),
    "XGBoost": XGBClassifier(n_estimators=100, learning_rate=0.05),
    "AdaBoost": AdaBoostClassifier(),
    "Extra Trees": ExtraTreesClassifier()
}

# Create a loop to test each meta-learner
for meta_learner_name, meta_learner in meta_learners.items():
    # Create the stacking ensemble with the current meta-learner
    stacking_ensemble = StackingClassifier(
        estimators=base_learners,
        final_estimator=meta_learner,
        cv=5
    )
"""
# Define a meta-learner
meta_learner = LogisticRegression()
# Create the stacking ensemble with the current meta-learner
stacking_ensemble = StackingClassifier(
    estimators=base_learners,
    final_estimator=meta_learner,
    cv=5
)
# Fit the ensemble
stacking_ensemble.fit(X_train, y_train)

# Fit the ensemble
stacking_ensemble.fit(X_train, y_train)

scoring_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
ensemble_results = EvaluateEnsemble(X_train, y_train, stacking_ensemble, scoring_metrics=scoring_metrics)
