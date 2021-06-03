# ------------------------------------------------------------------
#  Code to train a model to perform segmentation using Random Forest
#  and SVM classifiers
#  Autor: Rafael Vianna
#  Date: 2020 July
# -----------------------------------------------------------------

from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
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
df = pd.read_csv("dataset\heart_failure_clinical_records.csv",
                 sep=',', delimiter=None, header='infer')

# Normalization
# ------------------------------------------------------------------
scaler = MinMaxScaler()
scaler.fit(df.iloc[:, :-1])
df_Normalized = pd.DataFrame(scaler.transform(
    df.iloc[:, :-1]), columns=df.iloc[:, :-1].columns)

# ------------------------------------------------------------------
# ---------------------- Training and Validation -------------------
# ------------------------------------------------------------------

# Dependent variables
# ------------------------------------------------------------------
Y = df['DEATH_EVENT'].values

# Independent variables
# ------------------------------------------------------------------
X = df.drop(labels=['DEATH_EVENT'], axis=1)

# Classifiers
# ------------------------------------------------------------------
# Random Forest Classifier
rfc_model = RandomForestClassifier(n_estimators=10)
# Suport Vector Classifier
svc_model = LinearSVC(max_iter=100, tol=1e-2,
                      multi_class='crammer_singer', loss='hinge', C=1)
# Logistic RegressionClassifier
log_model = LogisticRegression(
    max_iter=100, tol=1e-2, solver='newton-cg', C=10)
# Naive Bayes Classifier
gnb_model = GaussianNB()


# Stratified 10-fold
# ------------------------------------------------------------------

accuracy_rfc = []
accuracy_svc = []
accuracy_log = []
accuracy_gnb = []
time_rfc = []
time_svc = []
time_log = []
time_gnb = []

skf = StratifiedKFold(n_splits=10)
skf.get_n_splits(X, Y)
for train_index, test_index in skf.split(X, Y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    Y_train, Y_test = np.ravel(pd.DataFrame(Y).iloc[train_index]), np.ravel(
        pd.DataFrame(Y).iloc[test_index])

    # RFC
    start = time.time()
    rfc_model.fit(X_train, Y_train)
    Y_pred = rfc_model.predict(X_test)
    score = accuracy_score(Y_pred, Y_test)
    accuracy_rfc.append(score)
    time_rfc.append(time.time()-start)

    # SVC
    start = time.time()
    svc_model.fit(X_train, Y_train)
    Y_pred = svc_model.predict(X_test)
    score = accuracy_score(Y_pred, Y_test)
    accuracy_svc.append(score)
    time_svc.append(time.time()-start)

    # Logistic Regression
    start = time.time()
    log_model.fit(X_train, Y_train)
    Y_pred = log_model.predict(X_test)
    score = accuracy_score(Y_pred, Y_test)
    accuracy_log.append(score)
    time_log.append(time.time()-start)

    # SVC
    start = time.time()
    gnb_model.fit(X_train, Y_train)
    Y_pred = gnb_model.predict(X_test)
    score = accuracy_score(Y_pred, Y_test)
    accuracy_gnb.append(score)
    time_gnb.append(time.time()-start)

print("RFC: ")
print("    Accuracy = ", np.array(accuracy_rfc).mean())
print("    Average Time = ", np.array(time_rfc).mean())
print("SVC: ")
print("    Accuracy = ", np.array(accuracy_svc).mean())
print("    Average Time = ", np.array(time_svc).mean())
print("Log_Reg: ")
print("    Accuracy = ", np.array(accuracy_log).mean())
print("    Average Time = ", np.array(time_log).mean())
print("N-B: ")
print("    Accuracy = ", np.array(accuracy_gnb).mean())
print("    Average Time = ", np.array(time_gnb).mean())
