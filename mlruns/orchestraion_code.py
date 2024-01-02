from typing import Any, Dict, List
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import mlflow
from prefect import task, flow


@task
def load_data(path):
    data = pd.read_csv(path)
    data['smoking_history'].replace({'never': 2, 'No Info': 3, 'current': 4, 'former': 5,
                                   'not current': 6, 'ever': 7}, inplace=True)
    data['gender'].replace({'Male': 2, 'Female': 3, 'Other': 3}, inplace=True)

    return data


@task
def get_classes(target_data):
    return list(target_data.unique())


@task
def get_scaler(data):
    scaler = StandardScaler()
    scaler.fit(data)
    return scaler


@task
def rescale_data(data, scaler):
    data_rescaled = pd.DataFrame(scaler.transform(data),
                                 columns=data.columns,
                                 index=data.index)
    return data_rescaled


@task
def split_data(input_, output_, test_data_ratio):
    X_tr, X_te, y_tr, y_te = train_test_split(input_, output_, test_size=test_data_ratio, random_state=0)
    return {'X_TRAIN': X_tr, 'Y_TRAIN': y_tr, 'X_TEST': X_te, 'Y_TEST': y_te}


@task
def find_best_model(X_train, y_train, estimator, parameters):
    # Enabling automatic MLflow logging for scikit-learn runs
    mlflow.sklearn.autolog(max_tuning_runs=None)

    with mlflow.start_run():
        clf = GridSearchCV(
            estimator=estimator,
            param_grid=parameters,
            scoring='accuracy',
            cv=5,
            return_train_score=True,
            verbose=1
        )
        clf.fit(X_train, y_train)

        # Disabling autologging
        mlflow.sklearn.autolog(disable=True)

        return clf


# Workflow
@flow
def main(path = 'C:\\Users\\TannaManohar\\Downloads\\diabetes_prediction_dataset.csv', target = 'diabetes',test_size = 0.2):
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Diabetes Prediction")

    # Define Parameters
    DATA_PATH = path
    TARGET_COL = target
    TEST_DATA_RATIO = test_size

    # Load the Data
    dataframe = load_data(path=DATA_PATH)

    # Identify Target Variable
    target_data = dataframe[TARGET_COL]
    input_data = dataframe.drop([TARGET_COL], axis=1)

    # Get Unique Classes
    classes = get_classes(target_data=target_data)

    # Split the Data into Train and Test
    train_test_dict = split_data(input_=input_data, output_=target_data, test_data_ratio=TEST_DATA_RATIO)

    # Rescaling Train and Test Data
    scaler = get_scaler(train_test_dict['X_TRAIN'])
    train_test_dict['X_TRAIN'] = rescale_data(data=train_test_dict['X_TRAIN'], scaler=scaler)
    train_test_dict['X_TEST'] = rescale_data(data=train_test_dict['X_TEST'], scaler=scaler)

    # Model Training
    ESTIMATOR = KNeighborsClassifier()
    HYPERPARAMETERS = [{'n_neighbors': [i for i in range(1, 51)], 'p': [1, 2]}]
    classifier = find_best_model(train_test_dict['X_TRAIN'], train_test_dict['Y_TRAIN'], ESTIMATOR, HYPERPARAMETERS)
    print(classifier.best_params_)
    print(classifier.score(train_test_dict['X_TEST'], train_test_dict['Y_TEST']))


Deploy the main function
from prefect.deployments import Deployment
from prefect.client.schemas.schedules import IntervalSchedule
from datetime import timedelta

deployment = Deployment.build_from_flow(
    flow=main,
    name="model_training",
    schedule=IntervalSchedule(interval=timedelta(minutes=5)),
    work_queue_name="ml"
)

deployment.apply()