# ------------------------------------------------------------------
#  Prediction of survival patients from heart failure dataset
#  Autor: Rafael Vianna
#  Date: 2021 June
# -----------------------------------------------------------------

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import time


# ------------------------------------------------------------------
# ---------------------- DataFrame Preparation ---------------------
# ------------------------------------------------------------------

# Import data
# ------------------------------------------------------------------
df = pd.read_csv("dataset/heart_failure_clinical_records.csv",
                 sep=',', delimiter=None, header='infer')
# Analyze dataset
# print(df.head)
# print(df.describe())

# Dependent variables
# ------------------------------------------------------------------
Y = df['DEATH_EVENT'].values

# Independent variables
# ------------------------------------------------------------------
# Normalization
scaler = MinMaxScaler()
scaler.fit(df.iloc[:, :-1])
df_Normalized = pd.DataFrame(scaler.transform(
    df.iloc[:, :-1]), columns=df.iloc[:, :-1].columns)

X = df_Normalized.drop(labels=['time'], axis=1)
X = X.drop(labels=['sex', 'anaemia', 'high_blood_pressure',
           'diabetes', 'smoking'], axis=1)

# ------------------------------------------------------------------
# ----------------------- Hyperparameter Tuning --------------------
# ------------------------------------------------------------------

# Define the hyperparameter values for each model
# ------------------------------------------------------------------
model_params = {
    'KNN': {
        'model': KNeighborsClassifier(),
        'params': {'n_neighbors': [3, 5, 7, 9, 11, 13], 'weights': ['uniform', 'distance'], 'algorithm': ['ball_tree', 'kd_tree', 'brute']}
    },
    'random_forest': {
        'model': RandomForestClassifier(),
        'params': {'n_estimators': [30, 50, 80, 100], 'criterion': ['gini', 'entropy']}
    },
    'naive_bayes': {
        'model': GaussianNB(),
        'params': {}
    },
    'logistic_regression': {
        'model': LogisticRegression(max_iter=100000000, tol=1e-4),
        'params': {'C': [0.5, 1, 5, 10, 15, 20],
                   'solver': ['sag', 'saga', 'lbfgs', 'newton-cg']
                   }
    },
    'SVC': {
        'model': LinearSVC(max_iter=100000000, tol=1e-4),
        'params': {'C': [0.5, 1, 5, 10, 15, 20],
                   'loss': ['hinge', 'squared_hinge'],
                   'multi_class': ['ovr', 'crammer_singer']
                   }
    },
    'MLP': {
        'model': MLPClassifier(),
        'params': {'max_iter': [1000, 10000, 100000000], 'hidden_layer_sizes': [30, 50, 100, 200, 500], 'activation': ['relu', 'logistic', 'tanh'], 'solver': ['lbfgs', 'sgd', 'adam'], 'learning_rate': ['constant', 'invscaling', 'adaptive']}
    }
}

# Create a grid search for each model
# ------------------------------------------------------------------
scores = []

for model_name, mp in model_params.items():
    print(model_name)
    start = time.time()
    clf = RandomizedSearchCV(
        mp['model'], mp['params'], cv=10, n_iter=20, n_jobs=-1)
    clf.fit(X, Y)
    scores.append({'model': model_name,
                   'best_score': clf.best_score_,
                   'best_params': clf.best_params_})
    print(model_name, "   ", "Time for parameter tuning = ", time.time()-start)

# ------------------------------------------------------------------
# ------------------------------ Results ---------------------------
# ------------------------------------------------------------------
result = pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])
print(result)

# Save Results
result.to_csv("results.csv")
