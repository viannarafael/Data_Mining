# ------------------------------------------------------------------
#  Code to compare classifiers for prediction of heart_failure_clinical_records
#  Autor: Rafael Vianna
#  Date: 2021 June
# -----------------------------------------------------------------

from sklearn.metrics import accuracy_score, recall_score, average_precision_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import time

# ------------------------------------------------------------------
# ---------------------- DataFrame Preparation ---------------------
# ------------------------------------------------------------------

# Import data
# ------------------------------------------------------------------
df = pd.read_csv("dataset\heart_failure_clinical_records.csv",
                 sep=',', delimiter=None, header='infer')

# ------------------------------------------------------------------
# ---------------------- Training and Validation -------------------
# ------------------------------------------------------------------

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
# X = X.drop(labels=['sex', 'anaemia', 'high_blood_pressure',
#            'diabetes', 'smoking'], axis=1)

# Classifiers
# ------------------------------------------------------------------
# K-NN
knn_model = KNeighborsClassifier(
    weights='distance', n_neighbors=7, algorithm='kd_tree')
# Random Forest Classifier
rfc_model = RandomForestClassifier(n_estimators=100, criterion='gini')
# Naive Bayes Classifier
gnb_model = GaussianNB()
# Logistic RegressionClassifier
log_model = LogisticRegression(
    max_iter=10000000, tol=1e-4, solver='sag', C=15)
# Suport Vector Classifier
svc_model = LinearSVC(max_iter=10000000, tol=1e-4,
                      multi_class='ovr', loss='squared_hinge', C=0.5)
# MLP Classifier
mlp_model = MLPClassifier(max_iter=10000, solver='adam',
                          learning_rate='invscaling', hidden_layer_sizes=200, activation='tanh')

# # Top 6 atributes
# # K-NN
# knn_model = KNeighborsClassifier(
#     weights='uniform', n_neighbors=11, algorithm='ball_tree')
# # Random Forest Classifier
# rfc_model = RandomForestClassifier(n_estimators=100, criterion='entropy')
# # Naive Bayes Classifier
# gnb_model = GaussianNB()
# # Logistic RegressionClassifier
# log_model = LogisticRegression(
#     max_iter=10000000, tol=1e-4, solver='saga', C=5)
# # Suport Vector Classifier
# svc_model = LinearSVC(max_iter=10000000, tol=1e-4,
#                       multi_class='ovr', loss='squared_hinge', C=1)
# # MLP Classifier
# mlp_model = MLPClassifier(max_iter=1000, solver='adam',
#                           learning_rate='constant', hidden_layer_sizes=500, activation='logistic')

# Stratified 10-fold
# ------------------------------------------------------------------
# Scoring
n = 10
scores_knn = np.zeros((5, n))
scores_rfc = np.zeros((5, n))
scores_gnb = np.zeros((5, n))
scores_log = np.zeros((5, n))
scores_svc = np.zeros((5, n))
scores_mlp = np.zeros((5, n))

# Split data
skf = StratifiedKFold(n_splits=n)
skf.get_n_splits(X, Y)
count = 0
for train_index, test_index in skf.split(X, Y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    Y_train, Y_test = np.ravel(pd.DataFrame(Y).iloc[train_index]), np.ravel(
        pd.DataFrame(Y).iloc[test_index])

# cross-validation for each model

    # KNN
    start = time.time()
    knn_model.fit(X_train, Y_train)
    Y_pred = knn_model.predict(X_test)
    scores_knn[0, count] = (accuracy_score(Y_pred, Y_test))
    scores_knn[1, count] = (recall_score(Y_pred, Y_test))
    scores_knn[2, count] = (average_precision_score(Y_pred, Y_test))
    scores_knn[3, count] = (f1_score(Y_pred, Y_test))
    scores_knn[4, count] = ((time.time()-start))

    # RFC
    start = time.time()
    rfc_model.fit(X_train, Y_train)
    Y_pred = rfc_model.predict(X_test)
    scores_rfc[0, count] = (accuracy_score(Y_pred, Y_test))
    scores_rfc[1, count] = (recall_score(Y_pred, Y_test))
    scores_rfc[2, count] = (average_precision_score(Y_pred, Y_test))
    scores_rfc[3, count] = (f1_score(Y_pred, Y_test))
    scores_rfc[4, count] = ((time.time()-start))
    #importances = list(model.feature_importances_)
    # feature_list = list(X.columns)
    # feature_imp = pd.Series(rfc_model.feature_importances_,
    #                         index=feature_list).sort_values(ascending=False)
    # print("\nFeature Relevance")
    # print(feature_imp)

    # GNB
    start = time.time()
    gnb_model.fit(X_train, Y_train)
    Y_pred = gnb_model.predict(X_test)
    scores_gnb[0, count] = (accuracy_score(Y_pred, Y_test))
    scores_gnb[1, count] = (recall_score(Y_pred, Y_test))
    scores_gnb[2, count] = (average_precision_score(Y_pred, Y_test))
    scores_gnb[3, count] = (f1_score(Y_pred, Y_test))
    scores_gnb[4, count] = ((time.time()-start))

    # LOG
    start = time.time()
    log_model.fit(X_train, Y_train)
    Y_pred = log_model.predict(X_test)
    scores_log[0, count] = (accuracy_score(Y_pred, Y_test))
    scores_log[1, count] = (recall_score(Y_pred, Y_test))
    scores_log[2, count] = (average_precision_score(Y_pred, Y_test))
    scores_log[3, count] = (f1_score(Y_pred, Y_test))
    scores_log[4, count] = ((time.time()-start))

    # SVC
    start = time.time()
    svc_model.fit(X_train, Y_train)
    Y_pred = svc_model.predict(X_test)
    scores_svc[0, count] = (accuracy_score(Y_pred, Y_test))
    scores_svc[1, count] = (recall_score(Y_pred, Y_test))
    scores_svc[2, count] = (average_precision_score(Y_pred, Y_test))
    scores_svc[3, count] = (f1_score(Y_pred, Y_test))
    scores_svc[4, count] = ((time.time()-start))

    # MLP
    start = time.time()
    mlp_model.fit(X_train, Y_train)
    Y_pred = mlp_model.predict(X_test)
    scores_mlp[0, count] = (accuracy_score(Y_pred, Y_test))
    scores_mlp[1, count] = (recall_score(Y_pred, Y_test))
    scores_mlp[2, count] = (average_precision_score(Y_pred, Y_test))
    scores_mlp[3, count] = (f1_score(Y_pred, Y_test))
    scores_mlp[4, count] = ((time.time()-start))

    count = count+1

# Summarize results and save to file
# ------------------------------------------------------------------
results = np.zeros((6, 5))
results[0, :] = np.transpose(np.mean(scores_knn, axis=1))
results[1, :] = np.transpose(np.mean(scores_rfc, axis=1))
results[2, :] = np.transpose(np.mean(scores_gnb, axis=1))
results[3, :] = np.transpose(np.mean(scores_log, axis=1))
results[4, :] = np.transpose(np.mean(scores_svc, axis=1))
results[5, :] = np.transpose(np.mean(scores_mlp, axis=1))

df = pd.DataFrame(results, columns=[
                  'Accuracy', 'Recall', 'Precision', 'F measure', 'Run Time'])
df['Classifier'] = ['K-NN', 'Randon Forest',
                    'Naive Bayes', 'Logistic', 'SVM', 'MLP']

# Save Results
df.to_csv("compare_classifier.csv")
