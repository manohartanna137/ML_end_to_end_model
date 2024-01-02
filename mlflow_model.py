import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import mlflow

df  = pd.read_csv("C:\\Users\\TannaManohar\\Downloads\\diabetes_prediction_dataset.csv")
df['smoking_history'].replace({'never': 2, 'No Info': 3, 'current': 4, 'former': 5,
                                'not current': 6, 'ever': 7}, inplace=True)
df['gender'].replace({'Male': 2, 'Female': 3, 'Other': 3}, inplace=True)
X = df.drop('diabetes',axis=1)
y = df['diabetes']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
scaler = StandardScaler()

# column names are (annoyingly) lost after Scaling
# (i.e. the dataframe is converted to a numpy ndarray)

X_train_rescaled = pd.DataFrame(scaler.fit_transform(X_train),
                                    columns = X_train.columns,
                                    index = X_train.index)

X_train_rescaled.head()

X_test_rescaled = pd.DataFrame(scaler.transform(X_test),
                                   columns = X_test.columns,
                                   index = X_test.index)

X_test_rescaled.head()



mlflow.set_experiment("Diabetes prediction")






mlflow.sklearn.autolog(max_tuning_runs=None)
with mlflow.start_run():
    mlflow.set_tag("dev", "Manohar")
    mlflow.set_tag("algo", "Logit")
    # log the data for each run using log_param, log_metric, log_model
    mlflow.log_param("data-path", "C:\\Users\\TannaManohar\\Downloads\\diabetes_prediction_dataset.csv")
    C = 0.1
    mlflow.log_param("C", C)
    lr_classifier = LogisticRegression(C=C)
    lr_classifier.fit(X_train_rescaled, y_train)
    y_test_pred = lr_classifier.predict(X_test_rescaled)
    acc = metrics.accuracy_score(y_test, y_test_pred)
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(lr_classifier, artifact_path="models")
    # mlflow.log_artifact("pickle/standard_scaler.pkl")

with mlflow.start_run():
    mlflow.set_tag("dev", "Manohar")
    mlflow.set_tag("algo", "DecisionTree")
    # log the data for each run using log_param, log_metric, log_model
    mlflow.log_param("data-path", "C:\\Users\\TannaManohar\\Downloads\\diabetes_prediction_dataset.csv")
    depth = 3
    mlflow.log_param("max_depth", depth)
    dt_classifier = DecisionTreeClassifier(max_depth = depth)
    dt_classifier.fit(X_train_rescaled, y_train)
    y_test_pred = dt_classifier.predict(X_test_rescaled)
    acc = metrics.accuracy_score(y_test, y_test_pred)
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(dt_classifier, artifact_path="models")
    # mlflow.log_artifact("pickle_files/standard_scaler.pkl")
with mlflow.start_run():
    mlflow.set_tag("dev", "")
    mlflow.set_tag("algo", "random forest")
    # log the data for each run using log_param, log_metric, log_model
    mlflow.log_param("data-path", "")
    param_grid = {
        'n_estimators': [25, 50, 100, 150],
        'max_features': ['sqrt', 'log2', None],
        'max_depth': [3, 6, 9],
        'max_leaf_nodes': [3, 6, 9],
    }

    grid_search = GridSearchCV(RandomForestClassifier(),
                               param_grid=param_grid)
    rf = grid_search.fit(X_train_rescaled, y_train)

    y_test_pred = rf.predict(X_test_rescaled)
    acc = metrics.accuracy_score(y_test, y_test_pred)
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(dt_classifier, artifact_path="models")
    # mlflow.log_artifact("pickle_files/standard_scaler.pkl")

