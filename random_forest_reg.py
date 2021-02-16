from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
import numpy as np


def forest_regression(traindata, labels, testdata):
    forest_reg = RandomForestRegressor
    param_grid = [{'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},]
    grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                               scoring='neg_mean_squared_error',
                               return_train_score=True)
    final_model = grid_search.best_estimator_
    # train model
    final_model.fit(traindata, labels)
    # evaluate model with test data
    x_test = testdata.drop("median_house_value", axis=1)
    y_test = testdata["median_house_value"].copy()
    x_test_prep = full_pipeline.transform(x_test)
    return

def display_scores(scores):
    print("Scores:", scores)
    print("mean:", scores.mean())
    print("Standard deviation:", scores.std())
    return
