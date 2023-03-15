import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import preprocessing
from category_encoders.ordinal import OrdinalEncoder
from category_encoders.woe import WOEEncoder
from category_encoders.target_encoder import TargetEncoder
from category_encoders.sum_coding import SumEncoder
from category_encoders.m_estimate import MEstimateEncoder
from category_encoders.leave_one_out import LeaveOneOutEncoder
from category_encoders.helmert import HelmertEncoder
from category_encoders.cat_boost import CatBoostEncoder
from category_encoders.james_stein import JamesSteinEncoder
from category_encoders.one_hot import OneHotEncoder
from sklearn.model_selection import  train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
sns.set()


def test_encoder(data, target):
    datasets = {"train": {}, "test": {}}
    datasets["train"]["data"], datasets["test"]["data"], datasets["train"]["target"], datasets["test"][
        "target"] = train_test_split(data, target, train_size=0.868055555556, shuffle=True)

    # Попробуем прогнать нашу же модель KNeighborsClassifier через подборку лучших гиперпараметров.

    parameters = {"n_neighbors": (3, 4, 5, 6, 7, 8, 9), "weights": ("uniform", "distance"),
                  "algorithm": ("ball_tree", "kd_tree", "brute"), "p": [1, 2]}
    clf = GridSearchCV(KNeighborsClassifier(), parameters)
    clf.fit(datasets["train"]["data"], datasets["train"]["target"])
    prediction_test = clf.predict(datasets["test"]["data"])

    # Считаем метрики
    mse = mean_squared_error(datasets["test"]["target"], prediction_test)
    accuracy = metrics.accuracy_score(datasets["test"]["target"], prediction_test)
    balanced_accuracy = metrics.balanced_accuracy_score(datasets["test"]["target"], prediction_test)
    precision = metrics.precision_score(datasets["test"]["target"], prediction_test, average='weighted')
    recall = metrics.recall_score(datasets["test"]["target"], prediction_test, average='weighted')

    print(f"Metrics for encoder {encoder}")
    print(f"MSE: {mse}")
    print("Presicion: ", round(precision, 2))
    print("Recall: ", round(recall, 2))
    print("Accurracy: ", round(accuracy, 2))
    print("Balanced: ", round(balanced_accuracy, 2))


cars = pd.read_csv('car.data', names=["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"])

target_encoder = preprocessing.LabelEncoder()
target = target_encoder.fit_transform(cars["class"])

cars.drop(columns=["class"], inplace=True)



cars_1 = cars.copy()
for column in cars_1:
    encoder = OrdinalEncoder()
    encoder.fit(cars_1[column])
    cars_1[column] = encoder.transform(cars_1[column])
print(cars_1)
test_encoder(cars_1, target)



cars_2 = cars.copy()
encoder = TargetEncoder()
encoder.fit(cars_2, target)
cars_2 = encoder.transform(cars_2)
print(cars_2)
test_encoder(cars_2, target)



cars_3 = cars.copy()
encoder = SumEncoder()
encoder.fit(cars_3, target)
cars_3 = encoder.transform(cars_3)
print(cars_3)
test_encoder(cars_3, target)


cars_4 = cars.copy()
encoder = MEstimateEncoder()
encoder.fit(cars_4, target)
cars_4 = encoder.transform(cars_4)
print(cars_4)
test_encoder(cars_4, target)

cars_5 = cars.copy()
encoder = MEstimateEncoder()
encoder.fit(cars_5, target)
cars_5 = encoder.transform(cars_5)
print(cars_5)
test_encoder(cars_5, target)
