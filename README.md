Note: In this tutorial, we will be using the Pima Indians Diabetes Database

https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database/data

Check the bottom for the acknowledgments

### Step 1: Research

There is often lots of information about common databases: best imputation methods, good models, etc.  If you are planning to really understand a database, do some research first.

#### i.e. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10048089/
Here is an analysis of different imputation methods with common classifiers

Really go out there and look around. What are people saying? How are they doing it? 

### Step 2: Understanding your task

From Kaggle : "The datasets consists of several medical predictor variables and one target variable, `Outcome`. Predictor variables includes the number of pregnancies the patient has had, their BMI, insulin level, age, and so on."

So this is a **binary classification** dataset, where we will be trying to predict whether a person will have diabetes. 

If you do a little search, you will find that these are some of the most common binary classification models (this will come in hand later) https://towardsdatascience.com/top-10-binary-classification-algorithms-a-beginners-guide-feeacbd7a3e2 by Alex Ortner:  
1. [Naive Bayes](https://scikit-learn.org/stable/modules/naive_bayes.html)
2. [Logistic Regression](https://towardsdatascience.com/understanding-logistic-regression-9b02c2aec102)
3. [K-Nearest Neighbors](https://towardsdatascience.com/machine-learning-basics-with-the-k-nearest-neighbors-algorithm-6a6e71d01761)
4. [Support Vector Machine](https://towardsdatascience.com/support-vector-machine-introduction-to-machine-learning-algorithms-934a444fca47)
5. [Decision Tree](https://towardsdatascience.com/decision-trees-in-machine-learning-641b9c4e8052)
6. [Bagging Decision Tree (Ensemble Learning I)](https://medium.com/ml-research-lab/bagging-ensemble-meta-algorithm-for-reducing-variance-c98fffa5489f)
7. [Boosted Decision Tree (Ensemble Learning II)](https://medium.com/ml-research-lab/boosting-ensemble-meta-algorithm-for-reducing-bias-5b8bfdce281)
8. [Random Forest (Ensemble Learning III)](https://medium.com/all-things-ai/in-depth-parameter-tuning-for-random-forest-d67bb7e920)
9. [Voting Classification (Ensemble Learning IV)](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html)
10. Deep Learning (We will be using Tensorflow)

### Step 3: Understanding the Data

Lets import some common libraries and check the head of the data:
```
import pandas as pd
import random

# Load the dataset
df = pd.read_csv('archive/diabetes.csv')

# Basics of Data Exploration
print(df.shape)

print(df.info())
```

![[Pasted image 20231217181252.png]]

While it may be obvious, this confirms it as a binary classification problem with 8 features and 1 target.

Let's check the head:

```
print(df.head())
```

This is what we get:

![[Pasted image 20231207153742.png]]

This seems weird. How does someone have 0 skin thickness and/or 0 Insulin? These are biologically impossible. If you do some research, it is most likely due to input errors and things of that nature.

Let's graph the data to get a better idea of what we are dealing with. **Note: the y-axis is the number of people**

```
from matplotlib import pyplot
pyplot.rcParams['figure.figsize'] = [20, 10]

df.hist()

pyplot.show()
```
![[Pasted image 20231207154320.png]]

We will definitely have to do something about the zeros for Glucose, BloodPressure, SkinThickness, Insulin, BMI

### Step 4: Preprocessing

One of the most important steps in the pipeline. We need to clean up the data to be better suited for learning. Based off the article from before, lets try Mice Imputation. 

```
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
  
# Replace zeros in specific columns with NaN

columns_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

df[columns_with_zeros] = df[columns_with_zeros].replace(0, np.nan)

  

# Apply MICE imputation

mice_imputer = IterativeImputer()

df_imputed = mice_imputer.fit_transform(df)

df_imputed_name=df.columns

# Convert the numpy array returned by MICE back to a pandas DataFrame

df_imputed = pd.DataFrame(df_imputed, columns=df.columns)

  

pyplot.rcParams['figure.figsize'] = [20, 10]

df_imputed.hist()

pyplot.show()

```
![[Pasted image 20231207160355.png]]

That's a lot better. However, there seem to still be some outliers in the data such as those in the DiabetesPedigreeFunction.

Lets use IQR:

```
#IQR method to detect outliers

Q1 = df_imputed.quantile(0.25)

Q3 = df_imputed.quantile(0.75)

IQR = Q3 - Q1

# Define a mask to identify rows with outliers

outlier_mask = ((df_imputed < (Q1 - 1.5 * IQR)) | (df_imputed > (Q3 + 1.5 * IQR))).any(axis=1)

# Filter out the outliers
filtered_df = df_imputed[~outlier_mask]

# Getting dataframe columns names
filtered_df_name=filtered_df.columns

pyplot.rcParams['figure.figsize'] = [20, 10]

filtered_df.hist()

pyplot.show()
```

![[Pasted image 20231208224615.png]]

You can play around the values when doing the IQR. You do not want to remove too much data, but outliers can skew your prediction as well.

Create a seed to allow for reproducibility

```
import random

SEED = 27
random.seed(SEED)
```

Lets split the data into a training set and a testing set **Note, X is uppercase and y is lowercase**

```
from sklearn.model_selection import train_test_split

# Split dataset into training set and test set

X = filtered_df[filtered_df_name[0:8]] # Features

Y = filtered_df[filtered_df_name[8]] # Target variable

X_train, X_test, y_train, y_test= train_test_split(X, Y, test_size=0.2, random_state=SEED, stratify=Y)
```

You might also notice that there is quite a class imbalance occurring in the outcomes. Let's fix that using SMOTE oversampling. Reminder that we do not want to give too much weight on the minority class, so you can play around with it to find a balance.

```
from imblearn.over_sampling import SMOTE

# Apply SMOTE to the training data to balance the dataset (oversampling)
pipeline = SMOTE(random_state=SEED)
X_train, y_train = pipeline.fit_resample(X_train, y_train)

# Lets check the number of outcomes in each class
print(pd.Series(y_train).value_counts())
```

![[Pasted image 20231217182204.png]]

Finally, let us use minmax scalar in order to normalize the features

```
from sklearn.preprocessing import MinMaxScaler

# Inspired by https://www.kaggle.com/code/pouryaayria/a-complete-ml-pipeline-tutorial-acu-86#Define-Problem:
# Minmax seems to do better than standard scaler

minmax_scaler = MinMaxScaler()
X_train = minmax_scaler.fit_transform(X_train)
X_test = minmax_scaler.transform(X_test)
```

### Step 5: Creating the Model and Tuning

Lets do some **hyperparameter tuning** with a diverse set of models

```
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier

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
```

```
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV

# Function to apply RandomizedSearchCV to each model

# Define cross-validation strategy
stratified_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)

# Function to apply RandomizedSearchCV to each model
def ApplyRandomizedSearch(models, param_dist, X_train, y_train, n_iter=100, cv=stratified_cv, random_state=SEED):
	"""
    Apply RandomizedSearchCV to a list of models with specified parameter distributions.
    Parameters:
    - models: List of (name, model) tuples
	- param_dist: Dictionary of parameter distributions {model_name:  distribution}
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
```

![[Pasted image 20231217183818.png]]
Note, the scores here are from the cross-validation scores averaged across the folds. The **RandomizedSearchCV** allows us to get parameters that are better generalized throughout the WHOLE dataset, not just the test dataset.

Lets instantiate the models based off the best hyperparameters from the validation. 
Note: Add verbose=-1 for Light Gradient Boost to suppress the training messages and probability=True and random_state=SEED for SVM. verbose=False for cat.

```
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
```

Fit and evaluate using a variety of scoring metrics

```
from sklearn.model_selection import cross_validate

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
    
scores = CVScoreModel(best_param_models, X_train, y_train, scoring_metrics=scoring_metrics, cv=stratified_cv)

print(scores)
```

![[Pasted image 20231218184026.png]]

Not bad.

## 6. Voting Ensemble and Correlation
Let's try a voting ensemble:

```
# Create a voting classifier for all models
estimator_model = [('NB', best_param_models['NB']), ('LR', best_param_models['LR']), ('KNN', best_param_models['KNN']), ('SVM', best_param_models['SVM']), ('CART', best_param_models['CART']), ('ada', best_param_models['ada']), ('gb', best_param_models['gb']), ('xgb', best_param_models['xgb']), ('lgb', best_param_models['lgb']), ('cat', best_param_models['cat']), ('rf', best_param_models['rf'])]
voting = VotingClassifier(estimators=estimator_model, voting='soft')

# Test on cross validation
scores = cross_validate(voting, X_train, y_train, scoring=scoring_metrics, cv=stratified_cv, n_jobs=-1)

# Calculating averages for each metric
average_scores = {metric: np.mean(scores[f'test_{metric}']) for metric in scoring_metrics}

# Print average scores
for metric, avg_score in average_scores.items():
    print(f"Average {metric}: {avg_score:.4f}")
```
![[Pasted image 20231217185502.png]]
Seems to be worse in terms of accuracy, lets see how the models correlate to each other

```
import plotly.graph_objs as go
import seaborn as sns
import matplotlib.pyplot as plt

# Train each model and collect their predictions
predictions = []

for name, model in best_param_models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)  # Assume X_val is your validation set
    predictions.append(preds)

  

# Calculate correlation matrix
# Convert predictions to a boolean format (correct or incorrect)
correct_predictions = [preds == y_test for preds in predictions]
correlation_matrix = np.corrcoef(correct_predictions)

# Visualize with heatmap
sns.heatmap(correlation_matrix, annot=True, xticklabels=[name for name, _ in best_param_models.items()], yticklabels=[name for name, _ in best_param_models.items()])
pyplot.title('Model Prediction Correlation')
pyplot.show()

```

Create map to check correlation between models:

![[Pasted image 20231218192119.png]]

Here's the part where you play around with it. While you cam do some better tuning, I just manually tested different values. I found that there was a low correlation between SVM and LR. However, it does introduce a bit overfitting due to the complexity of SVM. If you notice, ada seems to have decently low correlations in terms of the other two as well. Lastly, CAT seems to do pretty well by itself, let's see if we can add it to the ensemble to help with the robustness.

When playing with the weights, make sure to go very gradually (in small steps)

```
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
```

![[Pasted image 20231218200059.png]]

An accuracy of **~88%** and a roc_auc score of **~94%**

## Acknowledgments

1. Buczak, Philip et al. “Analyzing the Effect of Imputation on Classification Performance under 
		MCAR and MAR Missing Mechanisms.” _Entropy (Basel, Switzerland)_ vol. 25,3 521. 17 Mar. 2023, https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10048089/

2. Ortner, Alex. "Top 10 Binary Classification Algorithms a Beginner’s Guide." _Towards Data Science_, 
		28 May 2020, towardsdatascience.com/top-10-binary-classification-algorithms-a-beginners-guide-feeacbd7a3e2.

3. Pir. "A Complete ML Pipeline Tutorial: ACU 86." _Kaggle_, 2017, 
		[www.kaggle.com/code/pouryaayria/a-complete-ml-pipeline-tutorial-acu-86#Define-Problem](http://www.kaggle.com/code/pouryaayria/a-complete-ml-pipeline-tutorial-acu-86#Define-Problem)
